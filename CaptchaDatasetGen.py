
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

dir_captcha = "./dt/captcha"
dir_labels = "./dt/labels"

if os.path.exists(dir_labels):
    shutil.rmtree(dir_labels)
os.makedirs(dir_labels)

if os.path.exists(dir_captcha):
    shutil.rmtree(dir_captcha)
os.makedirs(dir_captcha)


def generate_captcha():
    img = Image.new('RGB', (640, 640), color='white')
    draw = ImageDraw.Draw(img)
    # text = ''.join([str(random.randint(0,9)) for _ in range(10)])
    text = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', k=8))

    draw.text((50, 100), text, fill='black', font=ImageFont.truetype('arial.ttf', 100))
    img.save(f'./dt/captcha/{text}.png')

    with open(f"./dt/labels/{text}.txt", 'w', encoding='utf-8') as f:
        f.write(f"{text}.png {text}")
    return text


import random
from captcha.image import ImageCaptcha

image = ImageCaptcha(width=200, height=100)


def generate():
    text = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', k=5))
    img = image.generate_image(text)
    img.save(f'./captcha/{text}.png')
    return text


for _ in range(10):
    generate_captcha()

    # generate()
