import os
from PIL import Image, ImageDraw, ImageFont
import random
import cv2
import numpy as np

# Path to store dataset
DATASET_PATH = r"D:\nrt_python_assignment\dataset"

# Document types
doc_types = ["invoice", "id_card", "certificate", "resume"]

# Sample text data
names = ["Ankit", "Sohan Panday", "Boby sumit", "Mohan Tiwari"]
amounts = ["₹2,500", "₹3,750", "₹1,200", "₹4,500"]
dates = ["2025-01-12", "2025-02-28", "2025-03-15", "2025-04-20"]
ids = ["AB123456", "XY987654", "CD654321", "EF112233"]

# Fonts folder (download some .ttf fonts here)
FONTS_FOLDER = r"D:\nrt_python_assignment\fonts"
fonts = [f for f in os.listdir(FONTS_FOLDER) if f.endswith(".ttf")]
print(fonts)

def create_text_image(doc_type, save_path, idx):
    # Image size
    width, height = 600, 400
    image = Image.new("RGB", (width, height), color=(255, 255, 255))

    draw = ImageDraw.Draw(image)

    # Random font
    font_files = [os.path.join(FONTS_FOLDER, f) for f in os.listdir(FONTS_FOLDER)]
    font_path = random.choice(font_files)
    font_size = random.randint(20, 30)
    font = ImageFont.truetype(font_path, font_size)

    # Add text based on doc_type
    if doc_type == "invoice":
        text = f"Name: {random.choice(names)}\nDate: {random.choice(dates)}\nAmount: {random.choice(amounts)}"
    elif doc_type == "id_card":
        text = f"Name: {random.choice(names)}\nID: {random.choice(ids)}"
    elif doc_type == "certificate":
        text = f"This certifies that {random.choice(names)} has completed the course."
    elif doc_type == "resume":
        text = f"{random.choice(names)}\nSkills: Python, ML, AI\nExperience: 2 Years"

    # Random position
    x = random.randint(20, 50)
    y = random.randint(20, 50)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # Add random noise: lines
    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=1)

    # Save image
    file_name = os.path.join(save_path, f"{doc_type}_{idx}.png")
    image.save(file_name)

def generate_dataset(num_images_per_type=50):
    for doc_type in doc_types:
        save_path = os.path.join(DATASET_PATH, doc_type)
        os.makedirs(save_path, exist_ok=True)
        for i in range(1, num_images_per_type + 1):
            create_text_image(doc_type, save_path, i)
            if i % 10 == 0:
                print(f"{doc_type}: {i} images generated")

if __name__ == "__main__":
    generate_dataset()
