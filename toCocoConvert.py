# import json
# from PIL import Image
# import os
#
# data = """220, 138, 598, 286, "Royal"
# 46, 275, 539, 390, "London"
# """
#
# lines = data.strip().split("\n")
#
# images = []
# annotations = []
# categories = {}
# cat_id_counter = 1
# ann_id_counter = 1
#
# images_folder = "data\\icdar2013\\TestImages"
#
# for img_id, line in enumerate(lines, start=1):
#     parts = line.split(", ")
#     x, y, w, h = map(int, parts[:4])
#     class_name = parts[4].strip('"')
#
#     image_path = os.path.join(images_folder, f"img_{img_id}.jpg")
#     with Image.open(image_path) as img:
#         width, height = img.size
#
#     if class_name not in categories:
#         categories[class_name] = cat_id_counter
#         cat_id_counter += 1
#
#     images.append({
#         "id": img_id,
#         "width": width,
#         "height": height,
#         "file_name": f"image_{img_id}.jpg"
#     })
#
#     annotations.append({
#         "id": ann_id_counter,
#         "image_id": img_id,
#         "bbox": [x, y, w, h],
#         "category_id": categories[class_name],
#         "area": w * h,
#         "iscrowd": 0
#     })
#     ann_id_counter += 1
#
# coco_format = {
#     "images": images,
#     "annotations": annotations,
#     "categories": [{"id": v, "name": k} for k, v in categories.items()]
# }
#
# # Save to file
# with open("annotations.json", "w") as f:
#     json.dump(coco_format, f, indent=2)
#
# print("COCO annotations saved to annotations.json")