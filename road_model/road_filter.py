import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the class colors
class_colors = [
    (0, 0, 0),       # background
    (255, 0, 0),     # building flooded
    (180, 120, 120), # building non-flooded --
    (160, 150, 20),  # road-flooded
    (140, 140, 140), # road-non-flooded
    (61, 230, 250),  # water --
    (0, 82, 255),    # tree
    (255, 0, 245),   # vehicle
    (255, 235, 0),   # pool
    (4, 250, 7)      # grass
]

def mask_road(image, dest_path=None):

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    target_colors = [
        (160, 150, 20),  # road-flooded
        (140, 140, 140)  # road-non-flooded
    ]

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for color in target_colors:
        color_mask = cv2.inRange(image, np.array(color) - 1, np.array(color) + 1)
        mask = cv2.bitwise_or(mask, color_mask)

    road_image = np.zeros_like(image)

    road_image[mask == 255] = image[mask == 255]

    filename = os.path.splitext(fname)[0]
    new_file = os.path.join(dest_path, filename+'.tiff')
    cv2.imwrite(new_file, road_image)

train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img-resized'
train_lab_path = '/home/anirud/Desktop/SemanticSeg/ColorMasks-TrainSet'

path='/home/anirud/Desktop/SemanticSeg/MaskPatchesOne'

files = os.listdir(path)
sorted_files = sorted(files)

for fname in sorted_files:
    img = os.path.join(path, fname)
    mask_road(img, dest_path='/home/anirud/Desktop/SemanticSeg/roads_only')
    