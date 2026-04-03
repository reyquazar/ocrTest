import os
from PIL import Image


def convert_folder():
    print("=" * 50)

    # 1. Запрашиваем папку с изображениями
    images_path = "coco8/images/" + input("Введите путь к папке с изображениями: ").strip()
    while not os.path.exists(images_path):
        print("Папка не найдена!")
        images_path = input("Введите путь к папке с изображениями: ").strip()

    # 2. Запрашиваем папку с аннотациями
    labels_path = "coco8/labels/" + input("Введите путь к папке с аннотациями (.txt): ").strip()
    while not os.path.exists(labels_path):
        print("Папка не найдена!")
        labels_path = input("Введите путь к папке с аннотациями: ").strip()

    # 3. Получаем список изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [f for f in os.listdir(images_path)
              if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"\nНайдено изображений: {len(images)}")

    # 4. Спрашиваем подтверждение
    confirm = input(f"\nКонвертировать аннотации для {len(images)} изображений? (y/n): ")
    if confirm.lower() != 'y':
        print("Отменено.")
        return

    print("\n" + "=" * 50)

    # 5. Конвертируем каждое изображение
    converted_count = 0
    error_count = 0

    for img_file in sorted(images):
        img_path = os.path.join(images_path, img_file)
        img_name = os.path.splitext(img_file)[0]

        # Ищем файл аннотации
        label_file = os.path.join(labels_path, f"{img_name}.txt")

        if not os.path.exists(label_file):
            print(f"⚠️  Пропущен: {img_file} → нет файла {img_name}.txt")
            error_count += 1
            continue

        try:
            # Узнаем размер изображения
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            print(f"\n📷 {img_file} | Размер: {img_width} x {img_height}")

            # Читаем аннотации
            with open(label_file, 'r') as f:
                lines = f.readlines()

            print(f"   Найдено аннотаций: {len(lines)}")

            # Конвертируем каждую строку
            yolo_annotations = []

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # Убираем кавычки и разбиваем
                cleaned = line.replace('"', '').replace('\"', '')
                parts = [p.strip() for p in cleaned.split(' ')]

                if len(parts) < 4:
                    print(f"   ⚠️  Строка {line_num}: неверный формат → пропущено")
                    continue

                try:
                    x1 = float(parts[0])
                    y1 = float(parts[1])
                    x2 = float(parts[2])
                    y2 = float(parts[3])

                    # class_id (0 по умолчанию)
                    class_id = 0

                    # Вычисляем YOLO координаты
                    box_width = x2 - x1
                    box_height = y2 - y1
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0

                    # Нормализуем
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = box_width / img_width
                    height_norm = box_height / img_height

                    # Проверка на валидность
                    if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1):
                        print(f"   ⚠️  Строка {line_num}: координаты выходят за пределы [0,1]")

                    yolo_annotations.append(
                        f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    )

                    print(
                        f"   Строка {line_num}: ({x1},{y1})→({x2},{y2}) → норм: {x_center_norm:.3f}, {y_center_norm:.3f}, {width_norm:.3f}, {height_norm:.3f}")

                except ValueError as e:
                    print(f"   ❌ Строка {line_num}: ошибка преобразования чисел - {e}")
                    continue

            # Сохраняем результат (перезаписываем файл)
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            print(f"   ✅ Сохранено {len(yolo_annotations)} аннотаций в {img_name}.txt")
            converted_count += 1

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            error_count += 1

    # Итог
    print("\n" + "=" * 50)
    print(f"✅ Успешно конвертировано: {converted_count}")
    print(f"⚠️  Ошибок/пропусков: {error_count}")
    print("=" * 50)


# Запуск
if __name__ == "__main__":
    convert_folder()
