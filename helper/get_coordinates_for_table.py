# Author: Xingyu Fu
# Date: 2025-07-07
# Description: This script is used to get the coordinates for the table cells in the Table vqa dataset.
# It is based on the code from the paper: ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding

import cv2
import numpy as np
import os
from tqdm import tqdm
import json
import random
random.seed(24)
import cv2
import numpy as np
import os
from tqdm import tqdm
import json
import random
random.seed(24)
from find_table import main as find_table_main
    

def get_table_boxes_cv2(input_image_path):
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find number of rows 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # cv2.drawContours(image, cnts, -1, (36,255,12), 2)
    rows_bbox = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(np.concatenate(c))
        rows_bbox.append({'x': x, 'y': y, 'w': w, 'h': h})
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 2)
    rows_bbox = sorted(rows_bbox, key = lambda x: x['y'])
    # Find number of columns
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # cv2.drawContours(image, cnts, -1, (36,255,12), 2)
    columns_bbox = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(np.concatenate(c))
        columns_bbox.append({'x': x, 'y': y, 'w': w, 'h': h})
    columns_bbox = sorted(columns_bbox, key = lambda x: x['x'])

    fixed_columns_bbox = {}
    fixed_rows_bbox = {}
    rows = len(rows_bbox) - 1
    columns = len(columns_bbox) -1
    if columns <= 0 or rows <= 0:
        height, width, _ = image.shape
        return (width, height, 0, 0), [], []
    else:
        left_x = min(columns_bbox + rows_bbox, key = lambda x: x['x'])['x']
        right_x = max(columns_bbox + rows_bbox, key = lambda x: x['x'])['x']
        top_y = min(columns_bbox + rows_bbox, key = lambda x: x['y'])['y']
        bottom_y = max(columns_bbox + rows_bbox, key = lambda x: x['y'])['y']

        # Draw Columns
        xs = list(set([b['x'] for b in columns_bbox] + [b['x'] + b['w'] for b in columns_bbox]))

        # Draw Rows
        ys = list(set([b['y'] for b in rows_bbox] + [b['y'] + b['h'] for b in rows_bbox]))
        return (left_x, top_y, right_x, bottom_y), xs, ys


def draw_table_boxes():
    # The minimum column width to be considered as a column
    min_text_width_limit = 30
    # The minimum row height to be considered as a row
    min_text_height_limit = 20
    # split = ''
    split = '-vwtq_syn'
    # split = '-vtabfact'
    data = json.load(open(f'data/tablevqa{split}.json')) # Input data
    # This draw_table_dir stores images with table coordinates drawn out for self-checking
    draw_table_dir = f'tablevqabench/images{split}_wCR'
    # This pre_draw_table_dir dir stores images with columns and rows drawn out for self-checking
    pre_draw_table_dir = f'tablevqabench/images{split}_wCR_pre'
    # This path stores the final bbox information
    bbox_path = f'data/tablevqa{split}_wbb.json'
    data_with_bbox = {}
    editable_count = 0
    mismatches = {'columns': 0, 'rows': 0}
    nested_column_header = 0
    for id, ex in tqdm(data.items()):
        input_image_path = ex['figure_path']
        image_name = os.path.basename(input_image_path)
        output_image_path = f'{draw_table_dir}/{image_name}'
        
        # Compare the two methods for finding the table boxes and combine the results
        (left_x, top_y, right_x, bottom_y), text_boxes = find_table_main(input_image_path, out_file=f'{pre_draw_table_dir}/{image_name}')
        (left_x_2, top_y_2, right_x_2, bottom_y_2), xs_2, ys_2 = get_table_boxes_cv2(input_image_path)

        # Get the main table coordinates from combining the two methods -- get the largest table
        if left_x_2 < left_x:
            left_x = left_x_2
        if top_y_2 < top_y:
            top_y = top_y_2
        if right_x_2 > right_x:
            right_x = right_x_2
        if bottom_y_2 > bottom_y:
            bottom_y = bottom_y_2 
        
        # Get the column coordinates as xs and row coordinates as ys
        fixed_columns_bbox = {}
        fixed_rows_bbox = {}
        image = cv2.imread(input_image_path)
        height, width, _ = image.shape
        last_x = None
        # Get all the detected x coordinates
        xs = list(set([b['x'] for b in text_boxes] + [b['x'] + b['w'] for b in text_boxes] + xs_2 + [left_x, right_x]))
        # Get all the detected y coordinates
        ys = list(set([b['y'] for b in text_boxes] + [b['y'] + b['h'] for b in text_boxes] + ys_2 + [top_y, bottom_y]))
        # remove duplicated x coordinates, make sure the column width is greater than min_text_width_limit
        for c in sorted(xs):
            if c not in fixed_columns_bbox:
                if not last_x:
                    last_x = c
                elif c - last_x >= min_text_width_limit:
                    cv2.rectangle(image,(last_x+1, top_y),(c-1, bottom_y),(0,255,0), 2)
                    fixed_columns_bbox[c] = {'x1': (last_x+1) / width, 'y1': top_y / height, 'x2': (c-1) / width, 'y2': bottom_y / height}
                    last_x = c
        # remove duplicated y coordinates, make sure the row height is greater than min_text_height_limit
        last_y = None
        for r in sorted(ys):
            if r not in fixed_rows_bbox:
                if not last_y:
                    last_y = r
                elif r - last_y >= min_text_height_limit:
                    cv2.rectangle(image,(left_x, last_y+1),(right_x, r-1),(255, 0,0), 2)
                    fixed_rows_bbox[r] = {'x1': left_x / width, 'y1': (last_y+1) / height, 'x2': right_x / width, 'y2': (r-1) / height}
                    last_y = r
        # Draw the columns and rows and save to pre_draw_table_dir
        cv2.imwrite(f'{pre_draw_table_dir}/{image_name}', image)

        # Reload the original image
        image = cv2.imread(input_image_path)
        # Check if the columns and rows are correct by comparing with the ground truth

        # Get the row and column numbers and headers from the ground truthhtml data
        text_html_table = ex['text_html_table'][:-1]
        rows = text_html_table.count('<tr>')
        row_starters = [r.split('>')[1].split('</')[0] for r in text_html_table.split('<tr>')[1:]]
        column_headers = []
        columns = 0
        for row in text_html_table.split('</tr>'):
            if row.count('<th>') > columns:
                columns = row.count('<th>')
                column_headers = [k.replace('</th>', '') for k in row.split('<th>')[1:]]
        
        # Check if the columns match the ground truth
        editable = 2
        if len(list(fixed_columns_bbox.values())) == columns:
            ex['columns_bbox'] = list(fixed_columns_bbox.values())
            ex['column_headers'] = column_headers
            for bbox in fixed_columns_bbox.values():
                cv2.rectangle(image,(int(bbox['x1'] * width), int(bbox['y1'] * height)),(int(bbox['x2'] * width), int(bbox['y2'] * height)),(0,255,0), 2)
        else:
            ex['columns_bbox'] = []
            ex['column_headers'] = []
            mismatches['columns'] += 1
            if 'rowspan' in text_html_table:
                nested_column_header += 1
            print(f'{id} Mismatch in column count: {len(list(fixed_columns_bbox.values()))} vs {columns}')
            editable -= 1

        # Check if the rows match the ground truth
        if  len(list(fixed_rows_bbox.values())) == rows + 1:
            fixed_rows_bbox = {k: v for k, v in fixed_rows_bbox.items() if k != ys[0]}
        if len(list(fixed_rows_bbox.values())) == rows:
            ex['rows_bbox'] = list(fixed_rows_bbox.values())
            ex['row_starters'] = row_starters
            for bbox in fixed_rows_bbox.values():
                cv2.rectangle(image,(int(bbox['x1'] * width), int(bbox['y1'] * height)),(int(bbox['x2'] * width), int(bbox['y2'] * height)),(255,0,0), 2)
        else:
            ex['rows_bbox'] = []
            ex['row_starters'] = []
            mismatches['rows'] += 1
            print(f'{id} Mismatch in row count: {len(list(fixed_rows_bbox.values()))} vs {rows}')
            editable -= 1
        if editable > 0:
            editable_count += 1
            cv2.imwrite(output_image_path, image)
        ex['figure_path_wCR'] = output_image_path
        ex['table_bbox'] = {'x1': left_x / width, 'y1': top_y / height, 'x2': right_x / width, 'y2': bottom_y / height}
        data_with_bbox[id] = ex
        
    with open(bbox_path, 'w') as f:
        json.dump(data_with_bbox, f, indent=4)
    print('Mismatches: ', mismatches, len(data))
    print('Editable:', editable_count)

if __name__ == '__main__':
    draw_table_boxes()