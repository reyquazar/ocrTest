import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ==================== ВАША CRNN МОДЕЛЬ ====================
class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))

        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))

        conv_relu(6)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)


# ==================== КОНВЕРТЕР ДЛЯ CTC ====================
class CTCLabelConverter:
    def __init__(self, characters):
        self.characters = characters
        # blank = 0, символы с 1
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
        self.idx_to_char[0] = '⁻'  # blank символ

    def encode(self, text):
        """Текст в последовательность индексов"""
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices, remove_repeats=True):
        """Индексы в текст"""
        if remove_repeats:
            result = []
            previous = None
            for idx in indices:
                if idx != 0 and idx != previous:
                    result.append(self.idx_to_char[idx])
                previous = idx
            return ''.join(result)
        return ''.join([self.idx_to_char[idx] for idx in indices if idx != 0])


# ==================== ДАТАСЕТ ====================
class TextDataset(Dataset):
    def __init__(self, num_samples=2000, img_height=32, img_width=128):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.images, self.labels = self._generate_data()

    def _generate_data(self):
        images = []
        labels = []

        for i in range(self.num_samples):
            # Создаем белое изображение
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255

            # Генерируем случайный текст (3-8 символов)
            text_len = np.random.randint(3, 8)
            text = ''.join(np.random.choice(list(self.characters), text_len))

            # Рисуем текст
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = np.random.uniform(0.5, 0.8)
            thickness = np.random.randint(1, 3)

            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = np.random.randint(5, max(6, self.img_width - text_w - 5))
            y = np.random.randint(text_h + 5, self.img_height - 5)

            cv2.putText(img, text, (x, y), font, font_scale, 0, thickness)

            # Инверсия (текст белый на черном)
            img = 255 - img

            # Нормализация
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(text)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # Добавляем канал: (1, height, width)
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        return img_tensor, label


def collate_fn(batch):
    """Коллация с паддингом до максимальной ширины"""
    images = []
    labels = []

    for img, label in batch:
        images.append(img)
        labels.append(label)

    # Паддинг изображений до одинаковой ширины
    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    padded_images = []
    for img in images:
        if img.shape[2] < max_width:
            pad_width = max_width - img.shape[2]
            padding = torch.zeros((1, img.shape[1], pad_width))
            padded = torch.cat([img, padding], dim=2)
        else:
            padded = img
        padded_images.append(padded)

    images = torch.stack(padded_images, dim=0)
    return images, labels


# ==================== ОБУЧЕНИЕ ====================
def train_crnn():
    # Параметры
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    NUM_CLASSES = len(CHARACTERS) + 1  # +1 для blank
    BATCH_SIZE = 16
    EPOCHS = 20

    # Создание датасета
    print("Генерация данных...")
    dataset = TextDataset(num_samples=2000, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Создание модели
    model = CRNN(
        img_channel=1,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_class=NUM_CLASSES,
        map_to_seq_hidden=64,
        rnn_hidden=256
    )

    converter = CTCLabelConverter(CHARACTERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            # Forward
            logits = model(images)  # (seq_len, batch, num_class)

            # CTC loss expects: (seq_len, batch, num_class)
            log_probs = F.log_softmax(logits, dim=2)

            # input_lengths: длина последовательности для каждого элемента в батче
            seq_len = logits.size(0)  # количество временных шагов
            input_lengths = torch.full((logits.size(1),), seq_len, dtype=torch.long)

            # target_lengths и targets
            all_targets = []
            target_lengths = []
            for label in labels:
                encoded = converter.encode(label)
                all_targets.extend(encoded)
                target_lengths.append(len(encoded))

            targets = torch.tensor(all_targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            # Вычисление loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

        # Тестирование на нескольких примерах
        if (epoch + 1) % 5 == 0:
            test_model(model, converter, dataset)

    # Сохранение
    torch.save(model.state_dict(), 'crnn_model.pth')
    print("✅ Модель сохранена как 'crnn_model.pth'")

    return model


def test_model(model, converter, dataset, num_samples=5):
    """Тестирование модели на случайных примерах"""
    model.eval()

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            img, true_label = dataset[idx]
            img_batch = img.unsqueeze(0)  # (1, 1, 32, 128)

            logits = model(img_batch)  # (seq_len, 1, num_class)
            _, preds = logits.max(2)
            preds = preds.squeeze(1).cpu().numpy()  # (seq_len,)

            predicted = converter.decode(preds)
            print(f"True: '{true_label}' -> Pred: '{predicted}'")

    model.train()


# ==================== ИНТЕГРАЦИЯ С YOLO ====================
class YOLO_CRNN_OCR:
    def __init__(self, yolo_model_path, crnn_model_path=None):
        # Загрузка YOLO
        self.yolo = YOLO(yolo_model_path)

        # Настройки CRNN
        self.img_height = 32
        self.img_width = 128
        self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.num_classes = len(self.characters) + 1

        # Инициализация CRNN
        self.crnn = CRNN(
            img_channel=1,
            img_height=self.img_height,
            img_width=self.img_width,
            num_class=self.num_classes
        )

        if crnn_model_path and os.path.exists(crnn_model_path):
            self.crnn.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
            print("✅ CRNN модель загружена")
        else:
            print("⚠️ CRNN модель не найдена")

        self.crnn.eval()
        self.converter = CTCLabelConverter(self.characters)

    def preprocess_region(self, roi):
        """Подготовка ROI для CRNN"""
        # Конвертация в серый
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Изменение размера
        resized = cv2.resize(gray, (self.img_width, self.img_height))

        # Бинаризация
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Инверсия (текст белый на черном)
        binary = 255 - binary

        # Нормализация
        normalized = binary.astype(np.float32) / 255.0

        return normalized

    def recognize_region(self, roi):
        """Распознавание текста в ROI"""
        try:
            # Предобработка
            processed = self.preprocess_region(roi)

            # Тензор
            img_tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0)

            # Распознавание
            with torch.no_grad():
                logits = self.crnn(img_tensor)  # (seq_len, 1, num_class)
                _, preds = logits.max(2)
                preds = preds.squeeze(1).cpu().numpy()
                text = self.converter.decode(preds)

            return text if text else "?"
        except Exception as e:
            return "ERR"

    def process_image(self, image_path, conf_threshold=0.5):
        """Полный пайплайн"""
        image = cv2.imread(image_path)
        if image is None:
            return []

        # Детекция YOLO
        results = self.yolo.predict(source=image_path, conf=conf_threshold, verbose=False)

        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    roi = image[y1:y2, x1:x2]

                    if roi.size > 0:
                        text = self.recognize_region(roi)
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'text': text,
                            'confidence': float(conf)
                        })

        return detections

    def visualize(self, image_path, detections, save_path=None):
        """Визуализация"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            text = det['text']
            conf = det['confidence']

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{text} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return image


# ==================== ОСНОВНОЙ ЗАПУСК ====================
def main():
    # Выбор режима
    print("=" * 50)
    print("1. Обучение CRNN")
    print("2. Использование YOLO + CRNN")
    choice = input("Выберите режим (1/2): ")

    if choice == '1':
        # Обучение модели
        train_crnn()

    elif choice == '2':
        # Загрузка YOLO
        print("Загрузка YOLO модели...")
        yolo_path = "yolo11-text.pt"

        # Создание OCR
        ocr = YOLO_CRNN_OCR(yolo_path, 'crnn_model.pth')

        # Обработка изображения
        img_path = "drive/MyDrive/icdar2013/TestImages/img_51.jpg"

        if os.path.exists(img_path):
            results = ocr.process_image(img_path)

            print("\nРЕЗУЛЬТАТЫ:")
            for i, r in enumerate(results, 1):
                print(f"{i}. '{r['text']}' - {r['bbox']}")

            # Визуализация
            vis = ocr.visualize(img_path, results, 'output.jpg')
            plt.figure(figsize=(15, 10))
            plt.imshow(vis)
            plt.axis('off')
            plt.show()
        else:
            print(f"Изображение не найдено: {img_path}")


if __name__ == "__main__":
    main()