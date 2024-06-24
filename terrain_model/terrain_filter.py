import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

target_colors = [
    (0, 0, 0), # background
    (255, 0, 0), # building flooded
    (180, 120, 120), #building non-flooded
    (61, 230, 250), # water
    (0, 82, 255), # tree
    (255, 0, 245), # vehicle
    (255, 235, 0), # pool
    (4, 250, 7) # grass
]

def mask_nonroad(image, fname=None, dest_path=None):

    if fname == None or dest_path == None:
        raise Exception("One or more arguments are None")

    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for color in target_colors:
        color_mask = cv2.inRange(image, np.array(color)-1, np.array(color)+1)
        mask = cv2.bitwise_or(mask, color_mask)

    road_image = np.zeros_like(image)

    road_image[mask == 255] = image[mask == 255]

    new_file = os.path.join(dest_path, fname+'.tiff')
    cv2.imwrite(new_file, road_image)

path = '/home/anirud/Desktop/SemanticSeg/MaskPatchesOne'

files = os.listdir(path)
sorted_files = sorted(files)

for fname in sorted_files:
    img = os.path.join(path, fname)
    name = os.path.splitext(fname)[0]
    mask_nonroad(img, fname=name, dest_path='/home/anirud/Desktop/SemanticSeg/terrain_model/terrain_only')

