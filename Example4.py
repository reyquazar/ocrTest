import os
import cv2
from pathlib import Path


def convert_to_yolo(gt_dir, img_dir, output_label_dir, split="train"):
    os.makedirs(output_label_dir, exist_ok=True)

    for gt_file in os.listdir(gt_dir):
        if not gt_file.endswith('.txt'):
            continue

        img_name = gt_file.replace('.txt', '.jpg')  # или .png — проверь расширение
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue
        h, w = img.shape[:2]

        label_path = os.path.join(output_label_dir, gt_file)

        with open(os.path.join(gt_dir, gt_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue

            try:
                x1 = int(parts[0].strip())
                y1 = int(parts[1].strip())
                x2 = int(parts[2].strip())
                y2 = int(parts[3].strip())

                center_x = ((x1 + x2) / 2.0) / w
                center_y = ((y1 + y2) / 2.0) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            except:
                continue

        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        else:
            open(label_path, 'w').close()


base_path = "data/icdar2013"
training_gt_path = "data/icdar2013/TrainingGT"
training_image_path = "data/icdar2013/TrainingImages"
test_gt_path = "data/icdar2013/TestGT"
test_image_path = "data/icdar2013/TestImages"


convert_to_yolo(
    gt_dir=training_gt_path,
    img_dir=training_image_path,
    output_label_dir=r"icdar2013_yolo\labels\train",
    split="train"
)

convert_to_yolo(
    gt_dir=test_gt_path,
    img_dir=test_image_path,
    output_label_dir=r"icdar2013_yolo\labels\val",
    split="val"
)
