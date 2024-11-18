import os
import random
from PIL import Image
from Configurations.Path import *


def compress_image(image_path, output_path, max_size=(800, 800)):
    with Image.open(image_path) as img:
        # make smaller but with proportions
        img.thumbnail(max_size)
        img.save(output_path, quality=85)


os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for label in os.listdir(source_dir):
    label_path = os.path.join(source_dir, label)

    if os.path.isdir(label_path):
        label_train_dir = os.path.join(train_dir, label)
        label_test_dir = os.path.join(test_dir, label)
        os.makedirs(label_train_dir, exist_ok=True)
        os.makedirs(label_test_dir, exist_ok=True)

        images = os.listdir(label_path)

        # shuffle for random division
        random.shuffle(images)

        # test = 20%
        test_size = int(0.2 * len(images))

        test_images = images[:test_size]
        train_images = images[test_size:]

        # copy and compress
        for img in test_images:
            img_path = os.path.join(label_path, img)
            compressed_img_path = os.path.join(label_test_dir, img)
            compress_image(img_path, compressed_img_path)

        for img in train_images:
            img_path = os.path.join(label_path, img)
            compressed_img_path = os.path.join(label_train_dir, img)
            compress_image(img_path, compressed_img_path)
