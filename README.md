# <img src="assets/icon.png" width="35" /> ReFocus

This repo contains codes for the paper "ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding" accepted at ICML 2025. 

[**🌐 Homepage**](https://zeyofu.github.io/ReFocus/) |[**📑 Paper**](https://arxiv.org/abs/2501.05452) |  [**🤗 Training Data**](https://huggingface.co/datasets/ReFocus/ReFocus_Data) | [**🔗 Trained Model**](https://huggingface.co/ReFocus/Trained_Model)


## 🔔News

 **🎉[2025-05-01]: ReFocus is accepted to [ICML2025](https://icml.cc/)! See you in Canada.**
 
 **🔥[2025-01-12]: Releasing the codes for ReFocus and collected [training data](https://huggingface.co/datasets/ReFocus/ReFocus_Data) and [finetuned model](https://huggingface.co/Fiaa/ReFocus).**

# Introduction

![Alt text](assets/teaser.png)

## Download Training Data
14k collected training data is uploaded on [Huggingface](https://huggingface.co/datasets/ReFocus/ReFocus_Data).
Complete raw data can be found in the [Huggingface Dataset Files](https://huggingface.co/datasets/ReFocus/ReFocus_Data/tree/main), where the training data is under `chartqa_vcot.zip` and `train_chartQA_*.zip`, with other files being testing data.


# ReFocus Prompting
We inherit most of the prompting code following [Visual SketchPad](https://visualsketchpad.github.io/)

## Installation
```
conda create -n refocus python=3.11
conda activate refocus

pip install pyautogen==0.3.0
pip install 'pyautogen[jupyter-executor]'
pip install Pillow joblib matplotlib opencv-python numpy networkx scipy datasets
```

## Quick Start
### Task Data
We preprocessed each task and put them into tasks. Download from this [Huggingface Dataset Files](https://huggingface.co/datasets/ReFocus/ReFocus_Data/tree/main) and put everything under `data`.

* Notice that the finetuning data is under `chartqa_vcot.zip` and `train_chartQA_*.zip`, with the rest being testing data.

### Run a Task
Set up your openAI key which is required to run ReFocus with GPT-4 models.

```
export OPENAI_API_KEY=<your_key>
```

Run code for each task to prompt with ReFocus.
```
python src/main_chartQA.py
python src/main_tablevqa.py
python src/main_charxiv.py
```

## ReFocus Finetuning
We follow the [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook/blob/main/md/04.Fine-tuning/FineTuning_Vision.md) for the supervised finetuning experiments. 

### Download the Finetuned Model
We release our best finetuned ReFocus model with full chain-of-thought data in this [HuggingFace Link](https://huggingface.co/ReFocus/Trained_Model).

This model is finetuned based on Phi-3.5-vision, and we used the following prompt during evaluation
```
<|image|>\n{question}\nThought:
```
To enforce the model to generate bounding box coordinates to refocus, you could try this prompt:
```
<|image_1|>\n{question}\nThought: The areas to focus on in the image have bounding box coordinates:
```

### Finetune Quickstart
Follow the [Phi3CookBook](https://github.com/microsoft/Phi-3CookBook/blob/main/md/04.Fine-tuning/FineTuning_Vision.md), clone it, and following its setting for a new finetuning environment. 

```
git clone https://github.com/microsoft/Phi-3CookBook.git

# create a new conda environment
conda create -n phi3v python=3.10
conda activate phi3v

# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# other libraries needed to run the example code
pip install -r requirements.txt

# (optional) flash attention -- Ampere+ GPUs (e.g., A100, H100)
pip install ninja
MAX_JOBS=32 pip install flash-attn==2.4.2 --no-build-isolation

# (optional) QLoRA -- Turing+ GPUs (e.g., RTX 8000)
pip install bitsandbytes==0.43.1
```
Move the file 
```
mv finetune_hf_trainer_chartqa_vcot.py Phi-3CookBook/code/04.Finetuning/vision_finetuning/
```

Then you could train the model
```
cd Phi-3CookBook/code/04.Finetuning/vision_finetuning

python -m torch.distributed.run --nproc_per_node=8 finetune_hf_trainer_chartqa_vcot.py --full_train --data_dir data/chartqa_vcot --bf16 --use_flash_attention --batch_size 48 --output_dir outputs/chartqa_vcot_loop --learning_rate 1e-6 --num_train_epochs 2 --output_bbox 1
```

## Coordinate Acquisition
In case you are interested, we share the code we used to acquire the table and chart coordinates in ReFocus. 
```
python helper/get_coordinates_for_chart.py
python helper/get_coordinates_for_table.py
```

