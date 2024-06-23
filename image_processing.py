import os
import cv2
import matplotlib.pyplot as plt


def create_patches(img_path, dest_direc=None, bgr=False, patch_size=256):
    if bgr:
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(img_path)
    h, w, _ = image.shape
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    patch_id = 0

    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            patch = image[top:top + patch_size, left:left + patch_size]

            patch_fname = os.path.join(dest_direc, f"{name}_{patch_id}.tiff")
            cv2.imwrite(patch_fname, patch)
            patch_id += 1

def resize_images(img_path, dest_path, img_size=(1024, 1024), gray_scale=False):
    for fname in os.listdir(img_path):
        if gray_scale:
            test_img = cv2.imread(img_path + '/' + fname, 0)
        else:
            test_img = cv2.imread(img_path+'/'+fname, cv2.IMREAD_UNCHANGED)
        test_img = cv2.resize(test_img, img_size)
        
        filename = os.path.splitext(fname)[0]
        new_file = os.path.join(dest_path, filename+'.tiff')
        cv2.imwrite(new_file, test_img)
       # cv2.imwrite()


# train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img'
# dest_train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img-resized'

# resize_images(train_img_path, dest_train_img_path)

train_img_path = '/home/anirud/Desktop/SemanticSeg/train-org-img-resized'
test_lab_path = '/home/anirud/Desktop/SemanticSeg/ColorMasks-TrainSet'

for fname in os.listdir(test_lab_path):
    img = os.path.join(test_lab_path, fname)
    create_patches(img, dest_direc='/home/anirud/Desktop/SemanticSeg/MaskPatches128', patch_size=128)
