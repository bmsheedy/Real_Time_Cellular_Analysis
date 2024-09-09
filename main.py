"""
---Real-Time Cellular Analysis YOLO Formatter---
-----------------------------------------------------------------------------------------------------------------------
Data formatting of malaria dataset for YOLOv8 model.
-----------------------------------------------------------------------------------------------------------------------
---Created by Brandon Sheedy---
"""

import os
import pandas as pd


data = pd.read_csv('labels.csv')

# Getting class labels and creating ids.
labels = data['class'].unique()
ids = {label: idx for idx, label in enumerate(labels)}
output_dir = 'annotations'
os.makedirs(output_dir, exist_ok=True)

# Writing annotations.
for index, row in data.iterrows():
    filename = row['filename']
    w = row['width']
    h = row['height']
    label = row['class']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    id = ids[label]

    # Normalize coords.
    x_center = (xmin + xmax) / (2 * w)
    y_center = (ymin + ymax) / (2 * h)
    box_width = (xmax - xmin) / w
    box_height = (ymax - ymin) / h

    # YOLO formating for annotations.
    annotation = f"{id} {x_center} {y_center} {box_width} {box_height}"
    annotation_filename = os.path.splitext(filename)[0] + '.txt'
    annotation_path = os.path.join(output_dir, annotation_filename)
    with open(annotation_path, 'a') as file:
        file.write(annotation + '\n')

# Creating class file.
classes_file = os.path.join(output_dir, '_classes.txt')
with open(classes_file, 'w') as file:
    for label, id in ids.items():
        file.write(f"{id} {label}\n")
