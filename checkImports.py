import numpy as np
import cv2



def _generate_data():
    images = []
    labels = []
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    NUM_CLASSES = len(CHARACTERS) + 1  # +1 для blank
    BATCH_SIZE = 16
    EPOCHS = 20

    for i in range(200):
        img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255

        # Генерируем случайный текст (3-8 символов)
        text_len = np.random.randint(3, 8)
        text = ''.join(np.random.choice(list(CHARACTERS), text_len))

        # Рисуем текст
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = np.random.uniform(0.5, 0.8)
        thickness = np.random.randint(1, 3)

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = np.random.randint(5, max(6, IMG_WIDTH - text_w - 5))
        y = np.random.randint(text_h + 5, IMG_HEIGHT - 5)

        cv2.putText(img, text, (x, y), font, font_scale, 0, thickness)

        img = 255 - img

        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(text)

    return images, labels

# Запускаем функцию
images, labels = _generate_data()

# Выводим информацию о сгенерированных данных
print(f"Сгенерировано {len(images)} изображений")
print(f"Размер одного изображения: {images[0].shape}")
print(f"Тип данных: {images[0].dtype}")
print(f"Диапазон значений: [{images[0].min():.2f}, {images[0].max():.2f}]")
print(f"\nПримеры текстов:")
for i in range(5):
    print(f"  {i+1}. {labels[i]}")

import os

# Создаем папку для сохранения
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

images, labels = _generate_data()

# Сохраняем каждое изображение
for i, (img, label) in enumerate(zip(images, labels)):
    # Конвертируем обратно в формат 0-255 для сохранения
    img_save = (img * 255).astype(np.uint8)
    filename = f"{output_dir}/img_{i:04d}_{label}.png"
    cv2.imwrite(filename, img_save)

print(f"Сохранено {len(images)} изображений в папку '{output_dir}'")