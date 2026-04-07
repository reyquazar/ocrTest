from PIL import Image, ImageDraw, ImageFont
import random



import os
import shutil

dir_name = "captcha"

# 1. Delete the directory if it exists
if os.path.exists(dir_name):
    shutil.rmtree(dir_name)  # Removes directory and all contents [21, 37]

# 2. Create the new directory
os.makedirs(dir_name)        # Creates the directory (and parents if needed) [18, 22]


def generate_captcha():
    img = Image.new('RGB', (120, 40), color='white')
    draw = ImageDraw.Draw(img)
    text = ''.join([str(random.randint(0,9)) for _ in range(4)])
    draw.text((10, 5), text, fill='black')
    img.save(f'./captcha/captcha_{text}.png')
    return text

import random
from captcha.image import ImageCaptcha  # pip install captcha

image = ImageCaptcha(width=200, height=100)
def generate():
    text = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', k=5))
    img = image.generate_image(text)
    img.save(f'./captcha/captcha_{text}.png')
    return text

for _ in range(10):
    # generate_captcha()
    generate()