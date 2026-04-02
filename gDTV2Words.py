import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse


def load_words_from_dataset(dataset_path):
    """Loads words from dataset file - reads text file, extracts words, filters by length"""
    words = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().split()[0] if line.strip() else None
                if word and len(word) >= 2:
                    words.append(word)
        print(f"✅ Loaded {len(words)} words from dataset")
        return words
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def add_document_noise(img_array):
    """Adds document-like noise"""
    h, w = img_array.shape[:2]

    if random.random() < 0.2:
        num_spots = random.randint(1, 5)
        for _ in range(num_spots):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            radius = random.randint(2, 10)
            intensity = random.randint(5, 20)
            cv2.circle(img_array, (x, y), radius, (intensity, intensity, intensity), -1)

    if random.random() < 0.15:
        num_lines = random.randint(1, 3)
        for _ in range(num_lines):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
            thickness = random.randint(1, 2)
            intensity = random.randint(10, 30)
            cv2.line(img_array, (x1, y1), (x2, y2), (intensity, intensity, intensity), thickness)

    if random.random() < 0.1:
        shadow_intensity = random.randint(5, 15)
        direction = random.choice(['top', 'bottom', 'left', 'right'])
        if direction == 'top':
            img_array[:h // 8, :] = np.clip(img_array[:h // 8, :] - shadow_intensity, 0, 255)
        elif direction == 'bottom':
            img_array[7 * h // 8:, :] = np.clip(img_array[7 * h // 8:, :] - shadow_intensity, 0, 255)
        elif direction == 'left':
            img_array[:, :w // 8] = np.clip(img_array[:, :w // 8] - shadow_intensity, 0, 255)
        else:
            img_array[:, 7 * w // 8:] = np.clip(img_array[:, 7 * w // 8:] - shadow_intensity, 0, 255)

    return img_array


def apply_document_effects(img):
    """Applies document scanning effects"""
    if random.random() < 0.3:
        blur_type = random.choice(['gaussian', 'defocus'])
        if blur_type == 'defocus':
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
        else:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.5)))

    if random.random() < 0.25:
        try:
            quality = random.randint(30, 85)
            from io import BytesIO
            output = BytesIO()
            img.save(output, format='JPG', quality=quality, optimize=True)
            img = Image.open(output)
            img = img.copy()
            output.close()
        except Exception as e:
            print(f"⚠️ PNG compression error: {e}")

    if random.random() < 0.1:
        try:
            skew = random.uniform(-0.1, 0.1)
            width, height = img.size
            xshift = abs(skew) * width
            new_width = width + int(xshift)
            img = img.transform((new_width, height), Image.AFFINE,
                                (1, skew, -xshift if skew > 0 else 0, 0, 1, 0))
        except Exception as e:
            print(f"⚠️ Skew error: {e}")

    return img


def apply_safe_augmentations(img_array):
    """Applies safe augmentations that won't destroy the image"""
    h, w = img_array.shape[:2]

    # Light Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.randint(1, 3), img_array.shape).astype('uint8')
        img_array = cv2.add(img_array, noise)

    # Light blur
    if random.random() < 0.2:
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # Color variations (safer version)
    if random.random() < 0.2:
        try:
            # Small brightness adjustment
            brightness = random.uniform(0.9, 1.1)
            img_array = np.clip(img_array.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Small contrast adjustment
            contrast = random.uniform(0.95, 1.05)
            mean = np.mean(img_array)
            img_array = np.clip((img_array.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"⚠️ Color adjustment error: {e}")

    return img_array


def generate_synthetic_data():
    """Main function - generates synthetic text images with document-like appearance"""
    output_dir = "./syntData"
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = './train_cleaned_ocr_perfect.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('train_number', type=int, help='Number of training images')
    parser.add_argument('val_number', type=int, help='Number of val images')
    parser.add_argument('--dataset', type=str, default=dataset_path, help='Path to dataset file')
    args = parser.parse_args()

    azerbaijani_words = load_words_from_dataset(args.dataset)

    fonts_dir = "./3"

    print(f"🔍 Find fonts: {fonts_dir}")
    print(f"📊 Total unique words: {len(azerbaijani_words)}")

    random.shuffle(azerbaijani_words)
    split_idx = int(0.8 * len(azerbaijani_words))

    train_words = azerbaijani_words[:split_idx]
    val_words = azerbaijani_words[split_idx:]

    print(f"📊 Words split: Train - {len(train_words)}, val - {len(val_words)}")

    available_fonts = []
    if os.path.exists(fonts_dir):
        for file in os.listdir(fonts_dir):
            if file.lower().endswith('.ttf'):
                font_path = os.path.join(fonts_dir, file)
                try:
                    available_fonts.append(font_path)
                    print(f"✅ Font loaded: {file}")
                except Exception as e:
                    print(f"❌ Error font load: {file}: {e}")
    else:
        print(f"❌ Dir with fonts: {fonts_dir}")
        return

    if not available_fonts:
        print("❌ Not available fonts!")
        return

    print(f"🎯 Used {len(available_fonts)} fonts")

    # Document backgrounds
    document_backgrounds = [
        'white', '#f8f8f8', '#f0f0f0', '#f5f5f5', '#fafafa',
        '#fffaf0', '#fdf5e6', '#fff8dc',
        '#f0fff0', '#f5fffa', '#f0f8ff'
    ]

    def create_text_image(word, is_double_word=False):
        """Creates basic text image with proper text rendering"""
        font_size = random.randint(20, 28) if not is_double_word else random.randint(18, 24)
        bg_color = random.choice(document_backgrounds)

        text_color_variants = [
            (0, 0, 0), (20, 20, 20), (30, 30, 30),
            (10, 10, 10), (15, 15, 15)
        ]
        text_color = random.choice(text_color_variants)

        try:
            font_path = random.choice(available_fonts)
            font = ImageFont.truetype(font_path, font_size)

            text_width = 320
            text_height = 48

            img = Image.new('RGB', (text_width, text_height), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Simple gradient background (occasionally)
            if random.random() < 0.05:
                for y in range(img.height):
                    shade = 245 + int(10 * (y / img.height))
                    for x in range(img.width):
                        img.putpixel((x, y), (shade, shade, shade))

            bbox = font.getbbox(word)
            text_actual_width = bbox[2] - bbox[0]
            text_actual_height = bbox[3] - bbox[1]

            # Ensure text fits
            if text_actual_width > text_width - 20:
                # Text too long, reduce font size
                reduction_factor = (text_width - 40) / text_actual_width
                font_size = max(14, int(font_size * reduction_factor))
                font = ImageFont.truetype(font_path, font_size)
                bbox = font.getbbox(word)
                text_actual_width = bbox[2] - bbox[0]
                text_actual_height = bbox[3] - bbox[1]

            x_offset = (text_width - text_actual_width) // 2
            y_offset = (text_height - text_actual_height) // 2

            # Ensure offsets are positive
            x_offset = max(10, x_offset)
            y_offset = max(5, y_offset)

            draw.text((x_offset, y_offset), word, font=font, fill=text_color)

            return img, word

        except Exception as e:
            print(f"❌ Error creating text image for '{word}': {e}")
            return None, None

    def generate_images(word_list, count, prefix):
        """Generates images for given word list"""
        labels = []
        single_word_count = int(count * 0.6)
        double_word_count = count - single_word_count

        print(f"📝 Generating {single_word_count} single-word and {double_word_count} double-word images for {prefix}")

        # Generate single word images
        for i in range(single_word_count):
            word = random.choice(word_list)

            try:
                img, final_text = create_text_image(word, is_double_word=False)
                if img is None:
                    continue

                # Basic augmentations
                rotation = random.randint(-3, 3)
                if rotation != 0:
                    img = img.rotate(rotation, expand=True, fillcolor=random.choice(document_backgrounds))

                # Apply document effects
                try:
                    img = apply_document_effects(img)
                except Exception as e:
                    print(f"⚠️ Document effects skipped: {e}")

                # Convert to array for OpenCV operations
                img_array = np.array(img)

                # Safe augmentations
                img_array = apply_safe_augmentations(img_array)

                # Add document noise
                try:
                    img_array = add_document_noise(img_array)
                except Exception as e:
                    print(f"⚠️ Document noise skipped: {e}")

                # Resize to final dimensions
                final_width, final_height = 320, 48
                img_resized = cv2.resize(img_array, (final_width, final_height), interpolation=cv2.INTER_LINEAR)

                filename = f"{prefix}_single_{i:09d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), img_resized)

                labels.append(f"{filename}\t{final_text}")

                if (i + 1) % 1000 == 0:
                    print(f"✅ {prefix} single-word: {i + 1}/{single_word_count}")

            except Exception as e:
                print(f"❌ {prefix} single-word gen error for word '{word}': {e}")
                continue

        # Generate double word images
        for i in range(double_word_count):
            word1 = random.choice(word_list)
            word2 = random.choice(word_list)

            # Avoid same words
            while word2 == word1:
                word2 = random.choice(word_list)

            # Combine words with separator
            separators = [' ', '  ', '   ', ' - ']
            separator = random.choice(separators)
            combined_text = word1 + separator + word2

            try:
                img, final_text = create_text_image(combined_text, is_double_word=True)
                if img is None:
                    continue

                # Less rotation for double words
                rotation = random.randint(-2, 2)
                if rotation != 0:
                    img = img.rotate(rotation, expand=True, fillcolor=random.choice(document_backgrounds))

                # Apply document effects
                try:
                    img = apply_document_effects(img)
                except Exception as e:
                    print(f"⚠️ Document effects skipped: {e}")

                # Convert to array for OpenCV operations
                img_array = np.array(img)

                # Safe augmentations (less frequently for double words)
                if random.random() < 0.5:
                    img_array = apply_safe_augmentations(img_array)

                # Add document noise
                try:
                    img_array = add_document_noise(img_array)
                except Exception as e:
                    print(f"⚠️ Document noise skipped: {e}")

                # Resize to final dimensions
                final_width, final_height = 320, 48
                img_resized = cv2.resize(img_array, (final_width, final_height), interpolation=cv2.INTER_LINEAR)

                filename = f"{prefix}_double_{i:09d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), img_resized)

                labels.append(f"{filename}\t{final_text}")

                if (i + 1) % 1000 == 0:
                    print(f"✅ {prefix} double-word: {i + 1}/{double_word_count}")

            except Exception as e:
                print(f"❌ {prefix} double-word gen error for words '{word1}', '{word2}': {e}")
                continue

        return labels

    print("🚀 Generating training data...")
    labels_train = generate_images(train_words, args.train_number, "train")

    print("🧪 Generating val data...")
    labels_val = generate_images(val_words, args.val_number, "val")

    with open(os.path.join(output_dir, "train_list.txt"), 'w', encoding='utf-8') as f:
        for label in labels_train:
            f.write(label + '\n')

    with open(os.path.join(output_dir, "val_list.txt"), 'w', encoding='utf-8') as f:
        for label in labels_val:
            f.write(label + '\n')

    unique_train_words = len(set([label.split('\t')[1] for label in labels_train]))
    unique_val_words = len(set([label.split('\t')[1] for label in labels_val]))

    print(f"\n🎉 Generation completed!")
    print(f"📊 Training: {len(labels_train)} images, {unique_train_words} unique texts")
    print(f"📊 Val: {len(labels_val)} images, {unique_val_words} unique texts")


generate_synthetic_data()
