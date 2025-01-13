"""
example for finetuning Phi-3-V on the ChartQA dataset using the Hugging Face Trainer API
Modified from Idefics-2 finetuning notebook:
https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing

Install dependencies:
    pip install transformers==4.38.1 \
        datasets \
        accelerate==0.30.1 \
        peft \
        Levenshtein \
        deepspeed==0.13.1
minimal run:
    torchrun --nproc_per_node=4 finetune_hf_trainer_docvqa.py
"""
import argparse
import json
import os
import random
from pathlib import Path

import Levenshtein
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from PIL import Image
random.seed(42)
import datetime
import re, copy
import numpy as np
import os
from openai import OpenAI
import time



# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


def create_dataset(data_dir, output_bbox=False):
    """
    chartqa dataset with visual chain of thought data
    inference time /// input: original image, query
    There are different setups:
    Settings: 
        Generate focus bbox or not. 
    0. train /// input: original image, query /// output: cot, answer;
    1. train /// input: original image, query /// output: focus bbox + cot, answer;
    """    
    train_dataset = []
    with open(os.path.join(data_dir, f'train.jsonl')) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            # Get the focus areas bounding box coordinates and add to the vcot text if needed
            if output_bbox and d['focus_areas']:
                focus_areas = d['focus_areas']
                bbox_text = f'The areas to focus on in the image have bounding box coordinates: {focus_areas}.'
                if len(d['conversations']) > 1:
                    cot_texts = f"{d['conversations'][0]['response']} {bbox_text} Looking at these areas, {d['conversations'][1]['response']}"
                else:
                    cot_texts = f"{d['conversations'][0]['response']} {bbox_text}"
            else:
                # Get the visual chain of thought (vcot) text from the conversations
                cot_texts = ' '.join([c['response'] for c in d['conversations']])
            cot_texts = re.sub(' +', ' ', cot_texts)
            if i == 0:
                print(f'Example vcot text: {cot_texts}')
                
            images = d['conversations'][0]['images']
            d['vcot'] = cot_texts
            d['images'] = images
            train_dataset.append(d)

    eval_dataset = []
    with open(os.path.join(data_dir, 'val.jsonl')) as f:
        all_ds = [json.loads(line) for line in f]
        # Select 30 to save time
        selected_ds = random.sample(all_ds, 30)
        for d in selected_ds:
            images = d['conversations'][0]['images']
            del d['conversations'][0]['images']
            d['images'] = images
            eval_dataset.append(d)

    test_dataset = []
    with open(os.path.join(data_dir, 'test.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            images = d['conversations'][0]['images']
            del d['conversations'][0]['images']
            d['images'] = images
            test_dataset.append(d)
            
    print(f'Dataset statistics: train={len(train_dataset)}, eval={len(eval_dataset)}, test={len(test_dataset)}')
    return train_dataset, eval_dataset, test_dataset


def dataset_get_shard(dataset, num_shards, index):
    if not 0 <= index < num_shards:
        raise ValueError("index should be in [0, num_shards-1]")
    indices = np.arange(index, len(dataset), num_shards)
    return [dataset[i] for i in indices]


def create_lora_config(rank, alpha_to_rank_ratio=2.0, dropout=0.0, freeze_vision_model=False):
    linear_modules = [
        # Phi language modules
        'qkv_proj',  # attention
        'o_proj',
        'down_proj',  # MLP
        'gate_up_proj',
        'lm_head',
    ]
    if not freeze_vision_model:
        vision_linear_modules = [
            # CLIP modules
            'q_proj',  # attention
            'k_proj',
            'v_proj',
            'out_proj',
            'fc1',  # MLP
            'fc2',
            # image projection
            'img_projection.0',
            'img_projection.2',
        ]
        linear_modules.extend(vision_linear_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=round(rank * alpha_to_rank_ratio),
        lora_dropout=dropout,
        target_modules=linear_modules,
        init_lora_weights='gaussian',
    )
    return lora_config


def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model


class ChartQADataCollator:
    def __init__(self, processor):
        self.processor = processor


    def __call__(self, examples):
        """
        chartqa dataset with visual chain of thought data
        inference time /// input: original image, query
        There are different setups:
        Settings: 
            Generate focus bbox or not. 
        0. train /// input: original image, query /// output: cot, answer;
        1. train /// input: original image, query /// output: focus bbox + cot, answer;
        {
            'id': f'{split}-{id}',
            'conversations': conversations,
            'focus_areas': focus_areas,
            'query': query,
            'answer': answer,
            'source': type,
            'x_values_bbox': x_values_bbox,
            'y_values_bbox': y_values_bbox,
            'figure_bbox': figure_bbox,
            'images': images,
            'vcot': vcot,
        }
        """
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        images = []
        for image_path in example['images']:
            img = Image.open(image_path)
            images.append(img.copy())
            img.close()

        image_tag_text = ''.join([f'<|image_{i}|>' for i in range(1, len(images) + 1)])
        # text_dict = example['conversations'][0]
        vcot = example['vcot']
        question = example['query']
        answer = example['answer']
        prompt_message = {
            'role': 'user',
            'content': f'{image_tag_text}\n{question}',
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'Thought: {vcot}\nAnswer: {answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, images, return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        # Do not add bos token to answer
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def acc(ground_truth, predicted_answers):
    assert len(ground_truth) == len(
        predicted_answers
    ), 'Length of ground_truth and predicted_answers must match.'

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = str(ground_truth[i])
        o_q_i = str(predicted_answers[i])
        if a_i == o_q_i:
            max_score = 1
        else:
            max_score = 0

        total_score += max_score

    return total_score / N


def acc_include(ground_truth, predicted_answers):
    # Whether the ground truth answer is included in the predicted answer, this is not accurate. We use GPT eval.
    assert len(ground_truth) == len(
        predicted_answers
    ), 'Length of ground_truth and predicted_answers must match.'

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = str(ground_truth[i])
        o_q_i = str(predicted_answers[i])
        if a_i in o_q_i:
            max_score = 1
        else:
            max_score = 0

        total_score += max_score

    return total_score / N


def average_gpt_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(
        predicted_answers
    ), 'Length of ground_truth and predicted_answers must match.'

    N = len(ground_truth)
    # total_score = gpt_eval_all(ground_truth, predicted_answers)
    total_score = []
    for i in tqdm(range(N)):
        a_i = str(ground_truth[i])
        o_q_i = str(predicted_answers[i])
        score = gpt_eval_one(a_i, o_q_i)
        total_score.append(score)

    return sum(total_score) / N, total_score

@torch.no_grad()
def evaluate(model, processor, eval_dataset, save_path=None, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    model.eval()
    answers_unique = []
    generated_texts_unique = []


    eval_dataset_shard = dataset_get_shard(eval_dataset, num_shards=world_size, index=rank)
    for example in tqdm(eval_dataset_shard, disable=(rank != 0) or disable_tqdm):

    # for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        # Phi-3-V currently only supports batch_size == 1
        # example = eval_dataset_shard[i]
        # example = eval_dataset[i]
        img = Image.open(example['images'][0])
        image = img.copy()
        img.close()
        question = example['query']
        answer = example['answer']
        # image = example['image']
        # text_dict = example['texts'][0]
        # question = text_dict['user']
        # answer = text_dict['assistant']
        
        answers_unique.append(answer)

        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}\nThought:',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(prompt, [image], return_tensors='pt').to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=1024, do_sample=False
        )

        generated_texts = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated_texts_unique.extend(generated_texts)

    generated_texts_unique = [g.strip().strip('.') for g in generated_texts_unique]

    # gather outputs from all ranks
    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)

    if rank == 0:
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers_unique': answers_unique,
                    'generated_texts_unique': generated_texts_unique,
                }
                json.dump(save_dict, f)
        # This evaluation is not accurate. We use GPT eval.
        anls = acc_include(
            ground_truth=answers_unique,
            predicted_answers=generated_texts_unique,
        )
        # Uncomment the following lines to use GPT evaluation
        # gpt_anls, gpt_scores = average_gpt_similarity(
        #     ground_truth=answers_unique,
        #     predicted_answers=generated_texts_unique,
        # )
        gpt_anls, gpt_scores = -1, [-1]
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers_unique': answers_unique,
                    'generated_texts_unique': generated_texts_unique,
                    'anls': anls,
                    'gpt_anls': gpt_anls,
                    'gpt_scores': gpt_scores,
                }
                json.dump(save_dict, f)
        return anls, gpt_anls
    return None, None


def patch_clip_for_lora(model):
    # remove unused parameters and then monkey patch
    def get_img_features(self, img_embeds):
        clip_vision_model = self.img_processor.vision_model
        hidden_states = clip_vision_model.embeddings(img_embeds)
        hidden_states = clip_vision_model.pre_layrnorm(hidden_states)
        patch_feature = clip_vision_model.encoder(
            inputs_embeds=hidden_states, output_hidden_states=True
        ).hidden_states[-1][:, 1:]
        return patch_feature

    image_embedder = model.model.vision_embed_tokens
    layer_index = image_embedder.layer_idx
    clip_layers = image_embedder.img_processor.vision_model.encoder.layers
    if layer_index < 0:
        layer_index = len(clip_layers) + layer_index
    del clip_layers[layer_index + 1 :]
    del image_embedder.img_processor.vision_model.post_layernorm
    image_embedder.get_img_features = get_img_features.__get__(image_embedder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        '--full_train', action='store_true', help='Use full training dataset (ChartQA)'
    )
    parser.add_argument('--data_dir', type=str, default='data/chartqa_vcot', help='Data directory')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, default='output/chartqa_vcot', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')
    parser.add_argument('--output_bbox', type=int, default=0)
    args = parser.parse_args()
    args.output_dir = f'{args.output_dir}/lr{args.learning_rate}_ep{args.num_train_epochs}_bb{args.output_bbox}'
    result_file = f'{args.output_dir}/test_after.json'
    if os.path.exists(result_file):
        print(f'Output file already exists: {result_file}')
        return
    # print(args)
    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    train_dataset, eval_dataset, test_dataset = create_dataset(args.data_dir, bool(args.output_bbox))
    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            use_qlora=args.use_qlora,
        )


    num_gpus = accelerator.num_processes
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // num_gpus
    if args.bf16:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # NOTE currently only supports batch_size == 1
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},  # NOTE important for LoRA
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None if args.use_lora else DS_CONFIG_DICT,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        ddp_find_unused_parameters=False,
    )

    data_collator = ChartQADataCollator(processor)

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not args.use_qlora:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = model.to(f'cuda:{local_rank}')
    anls, gpt_anls = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
    )
    if accelerator.is_main_process:
        print('Current Time: ', datetime.datetime.now())
        print('Saving outputs to:', out_path)
        print('Evaluating before fine-tuning on the validation set ...')
        print(f'Average accuracy before finetuning on the validation set: {anls}')
        print(f'Average GPT score before finetuning on the validation set: {gpt_anls}')
    
    if args.use_lora:
        patch_clip_for_lora(model)
        lora_config = create_lora_config(
            rank=args.lora_rank,
            alpha_to_rank_ratio=args.lora_alpha_ratio,
            dropout=args.lora_dropout,
            freeze_vision_model=args.freeze_vision_model,
        )
        model.add_adapter(lora_config)
        model.enable_adapters()

    if args.freeze_vision_model:
        model.model.vision_embed_tokens.requires_grad_(False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # eval after fine-tuning
    if args.use_lora:
        # first try to clear GPU memory
        del model
        del trainer
        __import__('gc').collect()
        torch.cuda.empty_cache()

        # reload the model for inference
        # this part also serves as an example of how to load a trained model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # Phi-3-V is originally trained in bf16 + flash attn
            # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
            torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
        )
        patch_clip_for_lora(model)
        model.load_adapter(training_args.output_dir)
    else:
        # for full finetuning, GPU memory can't be cleared (likely caused by deepspeed
        # https://github.com/microsoft/DeepSpeed/issues/3677)
        # so we don't reload the model
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=not args.bf16)

        # below is a sample code snippet to load fully-finetuned model
        # model = AutoModelForCausalLM.from_pretrained(
        #     training_args.output_dir,
        #     # Phi-3-V is originally trained in bf16 + flash attn
        #     # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        #     torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        #     trust_remote_code=True,
        #     _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
        # )

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')
    anls, gpt_anls = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
    )
    if rank == 0:
        print('Evaluating after fine-tuning on the validation set ...')
        print(f'Average accuracy after finetuning on the validation set: {anls}')
        print(f'Average GPT score after finetuning on the validation set: {gpt_anls}')
    
    anls, gpt_anls = evaluate(
        model,
        processor,
        test_dataset,
        save_path=out_path / 'test_after.json',
        disable_tqdm=not args.tqdm,
    )
    if rank == 0:
        print('Evaluating before fine-tuning on the test set ...')
        print(f'Average accuracy after finetuning on the test set: {anls}')
        print(f'Average GPT score after finetuning on the test set: {gpt_anls}')
        print('Current Time: ', datetime.datetime.now())

# GPT Evaluations
def match_score(prompt):
    retry_limit = 5
    eval_model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for retry in range(retry_limit):
        try:
            response = eval_model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            score = response.choices[0].message.content
            return int(score.strip())
        except Exception as e:
            time.sleep(1)
    print('Failed to evaluate the answer')
    print(prompt)
    return -1

def gpt_eval_one(solution, prediction):
    grading_query = f'''
    Rate my prediction given the correct answer, regardless of the answer format. 

    # Example
    # Prediction: 180 Meters
    # Answer: 80 Meters
    # Your Response: 0

    # Example
    # Question: What is the difference?
    # Prediction: 69%
    # Answer: 69
    # Your Response: 1

    # Example
    # Question: What is the increase?
    # Prediction: 3%
    # Answer: 0.03
    # Your Response: 1

    Prediction: {prediction}
    Answer: {solution}
    If you think the prediction is correct, return 1, otherwise return 0. Return 0 or 1 only.'''
    answer_score = match_score(grading_query)
    return answer_score

def get_scores(prompt):
    retry_limit = 5
    eval_model = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for retry in range(retry_limit):
        try:
            response = eval_model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4095,
            )
            scores = response.choices[0].message.content
            print(scores)
            score_list = [int(s.strip()) for s in scores.split(',')]
            return score_list
        except Exception as e:
            time.sleep(1)
    print('Failed to evaluate the answer')
    print(prompt)
    return [-1]

def gpt_eval_all(solution, prediction):
    grading_query = f'''Rate my predictions given the correct answers, regardless of the answer format. 
If you think the prediction is correct, score it 1, otherwise score it 0. Return a list of 0 or 1 only.

# Example
# Prediction: ["180 Meters", "69%", "3%"]
# Answer: ["80 Meters", "69", "0.03"]
# Your Response: 0,1,1

Prediction: {prediction}
Answer: {solution}
Your Response: 
'''
    answer_scores = get_scores(grading_query)
    return answer_scores

if __name__ == '__main__':
    main()
