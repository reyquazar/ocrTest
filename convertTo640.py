import os

for split in ['train', 'val', 'test']:
    labels_dir = f'coco8/labels/{split}'
    for filename in os.listdir(labels_dir):
        if filename.startswith('gt_'):
            new_name = filename[3:]  # убираем 'gt_'
            old_path = os.path.join(labels_dir, filename)
            new_path = os.path.join(labels_dir, new_name)
            os.rename(old_path, new_path)
            print(f'{filename} → {new_name}')