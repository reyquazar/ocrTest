import os

# Пути к папкам
img_dir = "coco8/images/train"
lbl_dir = "coco8/labels/train"

# Получаем имена без расширений
img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
lbl_names = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}

# Какие изображения без разметки?
missing_labels = img_names - lbl_names
print(f"Изображений без разметки: {len(missing_labels)}")
print(list(missing_labels)[:10])  # первые 10

# Какая разметка без изображений?
missing_images = lbl_names - img_names
print(f"Файлов разметки без изображений: {len(missing_images)}")

# Если разница только в расширении — переименуем
if len(missing_labels) > 0 and all('png' in str(f).lower() for f in missing_labels):
    # Пример: в images .jpg, а в labels .png
    for lbl_file in os.listdir(lbl_dir):
        if lbl_file.endswith('.png.txt'):  # странный двойной?
            new_name = lbl_file.replace('.png.txt', '.txt')
            os.rename(os.path.join(lbl_dir, lbl_file),
                      os.path.join(lbl_dir, new_name))
