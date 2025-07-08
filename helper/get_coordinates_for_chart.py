# Author: Xingyu Fu
# Date: 2025-07-07
# Description: This script is used to get the coordinates for the chart figure coordinates in the ChartQA dataset.
# It is based on the code from the paper: ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding

import cv2
import os
from tqdm import tqdm
import json
import random
random.seed(24)
from collections import defaultdict
    
def get_xys(annotation):
    # Get the x and y axis values and bboxes from the ground truth annotation
    x_axis_values, x_axis_bbox = [], []
    y_axis_values, y_axis_bbox = [], []
    try:
        if 'x_axis' in annotation['general_figure_info']:
            x_axis_values, x_axis_bbox = annotation['general_figure_info']['x_axis']['major_labels']['values'], annotation['general_figure_info']['x_axis']['major_labels']['bboxes']
    except:
        pass
    try:
        if 'y_axis' in annotation['general_figure_info']:
            y_axis_values, y_axis_bbox = annotation['general_figure_info']['y_axis']['major_labels']['values'], annotation['general_figure_info']['y_axis']['major_labels']['bboxes']
    except:
        pass
    return x_axis_values, x_axis_bbox, y_axis_values, y_axis_bbox

def get_y_bbox_central(y_raw_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height):
    # Get the area bboxes given the y axis bboxes and the figure bbox
    y_bboxes = []
    y_raw_bbox = sorted(y_raw_bbox, key = lambda x: x['y'])
    last_y = figure_y1
    for i, bb in enumerate(y_raw_bbox):
        center_y = int(bb['y'] + bb['h'] / 2)
        if i == len(y_raw_bbox) - 1:
            half_y_height = figure_y2 - center_y
        else:
            half_y_height = (y_raw_bbox[i+1]['y'] - center_y) / 2
        y1 = max(0, int(center_y - half_y_height))
        y1 = max(last_y, y1)
        y2 = min(figure_y2, int(center_y + half_y_height))
        bbox = {'x1': figure_x1 / width, 'y1': y1 / height, 'x2': figure_x2 / width, 'y2': y2 / height}
        y_bboxes.append(bbox)
        last_y = y2
    return y_bboxes

def get_y_bbox_upward(y_raw_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height):
    y_bboxes = []
    y_raw_bbox = sorted(y_raw_bbox, key = lambda x: x['y'])
    last_y = figure_y1
    for i, bb in enumerate(y_raw_bbox):
        y1 = last_y
        y2 = int(bb['y'] + bb['h'] / 2)
        bbox = {'x1': figure_x1 / width, 'y1': y1 / height, 'x2': figure_x2 / width, 'y2': y2 / height}
        y_bboxes.append(bbox)
        last_y = y2
    return y_bboxes

def get_x_bbox_toleft(x_raw_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height):
    x_bboxes = []
    x_raw_bbox = sorted(x_raw_bbox, key = lambda x: x['x'])
    last_x = x_raw_bbox[0]['x'] - 5
    for i, bb in enumerate(x_raw_bbox):
        x1 = last_x
        x2 = int(bb['x'] + bb['w'] / 2)
        bbox = {'x1': x1 / width, 'y1': figure_y1 / height, 'x2': x2 / width, 'y2': figure_y2 / height}
        x_bboxes.append(bbox)
        last_x = x2
    return x_bboxes

def get_x_bbox_central(x_raw_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height):
    x_bboxes = []
    x_raw_bbox = sorted(x_raw_bbox, key = lambda x: x['x'])
    figure_y2 = max([int(bb['y'] + bb['h']) for bb in x_raw_bbox])
    last_x = figure_x1
    for i, bb in enumerate(x_raw_bbox):
        center_x = int(bb['x'] + bb['w'] / 2)
        if i == len(x_raw_bbox) - 1:
            half_x_width = figure_x2 - center_x
        else:
            half_x_width = (x_raw_bbox[i+1]['x'] - center_x) / 2
        x1 = max(last_x, int(center_x - half_x_width))
        x2 = min(figure_x2, int(center_x + half_x_width))
        bbox = {'x1': x1 / width, 'y1': figure_y1 / height, 'x2': x2 / width, 'y2': figure_y2 / height}
        x_bboxes.append(bbox)
        last_x = x2
    return x_bboxes

def draw_chart_plot():
    data = json.load(open(f'data/chartQA.json'))
    split = ['h_bar', 'v_bar']
    # This draw_chart_dir stores images with chart coordinates drawn out for self-checking
    draw_chart_dir = {type: f'data/ChartQA Dataset/test/images_{type}_wPlot' for type in split}
    # Save bounding box information to this path, _wbb means with bounding box
    output_path = {'h_bar': f'data/chartQA_h_bar_wbb.json', 'v_bar': f'data/chartQA_v_bar_wbb.json'}
    data_with_bbox = {'h_bar': {}, 'v_bar': {}}
    for id, ex in tqdm(data.items()):
        annotation_path = ex['annotation_path']
        annotation = json.load(open(annotation_path))
        type = annotation['type']
        if type not in split: continue
        image_input_path = ex['figure_path']
        image_name = os.path.basename(image_input_path).replace('.png', '.jpg')
        output_image_path = f'{draw_chart_dir[type]}/{image_name}'
        image = cv2.imread(image_input_path)
        height, width, _ = image.shape

        if ('general_figure_info' not in annotation) or ('x_axis' not in annotation['general_figure_info'] and 'y_axis' not in annotation['general_figure_info']):
            continue
        try:
            # Load ground truth information
            figure_bbox = annotation['general_figure_info']['figure_info']['bbox']
            figure_x1, figure_y1, figure_x2, figure_y2 = figure_bbox['x'], figure_bbox['y'], int(figure_bbox['x'] + figure_bbox['w']), int(figure_bbox['y'] + figure_bbox['h'])
            cv2.rectangle(image,(figure_x1, figure_y1), (figure_x2, figure_y2), (0, 0,255), 2)
            ex['figure_bbox'] = {'x1': figure_x1 / width, 'y1': figure_y1 / height, 'x2': figure_x2 / width, 'y2': figure_y2 / height}
        except:
            continue
        # Get the x and y axis values and bboxes from the ground truth annotation
        x_axis_values, x_axis_bbox, y_axis_values, y_axis_bbox = get_xys(annotation)
        x_values, x_bboxes, y_values, y_bboxes = [], [], [], []
        if type == 'h_bar':
            # If the figure is a horizontal bar chart, get the y coordinates for each bar
            figure_x1 = 5
            figure_x2 = width - 5
            # x and y are reversed
            if x_axis_values:
                y_values = x_axis_values
                y_raw_bbox = x_axis_bbox
            y_bboxes = get_y_bbox_central(y_raw_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height)

            ex['x_intervals'] = []
            ex['y_intervals'] = []
            ex['x_values'] = x_values
            ex['y_values'] = y_values
            ex['x_bboxes'] = x_bboxes
            ex['y_bboxes'] = y_bboxes
        if type == 'v_bar':
            # If the figure is a vertical bar chart, get the x coordinates for each bar
            x_values = x_axis_values
            x_bboxes = get_x_bbox_central(x_axis_bbox, figure_x1, figure_x2, figure_y1, figure_y2, width, height)
        # Draw the y coordinates for each bar in red
        for bb in y_bboxes:
            cv2.rectangle(image, (int(bb['x1'] * width), int(bb['y1'] * height)), (int(bb['x2'] * width), int(bb['y2'] * height)), (255, 0, 0), 2)
        # Draw the x coordinates for each bar in green
        for bb in x_bboxes:
            cv2.rectangle(image, (int(bb['x1'] * width), int(bb['y1'] * height)), (int(bb['x2'] * width), int(bb['y2'] * height)), (0, 255, 0), 2)
            ex['x_intervals'] = []
            ex['y_intervals'] = []
            ex['x_values'] = x_values
            ex['y_values'] = y_values
            ex['x_bboxes'] = x_bboxes
            ex['y_bboxes'] = y_bboxes
        # Save the image with the coordinates drawn out, for self-checking
        cv2.imwrite(output_image_path, image)

        ex['figure_path_wCR'] = output_image_path
        if 'x_values' in ex or 'y_values' in ex:
            data_with_bbox[type][id] = ex
    for type in split:
        with open(output_path[type], 'w') as f:
            json.dump(data_with_bbox[type], f, indent=4)
        print(f'Editable {type}:', len(data_with_bbox[type]))


if __name__ == '__main__':
    draw_chart_plot()
