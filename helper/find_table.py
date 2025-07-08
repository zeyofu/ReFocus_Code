import os
import cv2
import numpy as np

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns


def pre_process_image(img, save_in_file=None, morph_size=(2, 2)):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def find_text_boxes(pre, min_text_height_limit=20, max_text_height_limit=200, min_text_width_limit=30):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    xs = []
    ys = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        xs.append(box[0])
        xs.append(box[0] + box[2])
        ys.append(box[1])
        ys.append(box[1] + box[3])
    # select x and y that occur over 5 times
    filtered_xs = [i for i in set(xs) if xs.count(i) > 5]
    filtered_ys = [i for i in set(ys) if ys.count(i) > 5]
    for contour in contours:
        box = cv2.boundingRect(contour)
        x, y, w, h = box[0], box[1], box[2], box[3]

        if min_text_height_limit < h and w > min_text_width_limit:
            if x in filtered_xs and y in filtered_ys and x + w in filtered_xs and y + h in filtered_ys:
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h})

    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def find_table(pre, width, height, min_text_height_limit=6, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    min_x = width
    min_y = height
    max_x = 0
    max_y = 0
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            (x, y, w, h) = box
            x2 = x + w
            y2 = y + h
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x2 > max_x:
                max_x = x2
            if y2 > max_y:
                max_y = y2

    return min_x, min_y, max_x, max_y


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines


def main(in_file, out_file=None, pre_file=None):
    img = cv2.imread(os.path.join(in_file))
    height, width, channels = img.shape

    pre_processed = pre_process_image(img, save_in_file=pre_file)
    x1, y1, x2, y2 = find_table(pre_processed, width, height, max_text_height_limit=height)
    text_boxes = find_text_boxes(pre_processed, max_text_height_limit=height)
    if out_file:
        vis = img.copy()
        if (x1, y2, x2-x1, y2-y1) not in text_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        for box in text_boxes:
            x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imwrite(out_file, vis)
    
    return (x1, y1, x2, y2), text_boxes
