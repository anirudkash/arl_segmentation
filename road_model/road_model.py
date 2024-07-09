import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from unet_model import UNet, jacard_coef
from keras.utils import normalize
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


class_colors = [
    (0, 0, 0), # background
    (160, 150, 20), # road-flooded
    (140, 140, 140) # road-non-flooded

]

background = np.array(class_colors[0])
road_flooded = np.array(class_colors[1])
road_nonflooded = np.array(class_colors[2])

num_classes = len(class_colors)

train_imgs_direc = '/home/anirud/Desktop/SemanticSeg/TrainImgsOne'
train_imgs_labs_direc = '/home/anirud/Desktop/SemanticSeg/road_model/roads_only'

def load_images(img_direc, gt_direc):

    train_imgs = []
    train_imgs_labs = []

    train_imgs_entries = os.listdir(img_direc)
    train_imgs_entries.sort()

    train_imgs_labs_entries = os.listdir(gt_direc)
    train_imgs_labs_entries.sort()

    for fname in train_imgs_entries:
        path = os.path.join(img_direc, fname) 
        # might have to read in the image as gray scale 
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        train_imgs.append(img)

    for fname in train_imgs_labs_entries:
        path = os.path.join(gt_direc, fname)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
       # img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        train_imgs_labs.append(img)

    train_images = np.array(train_imgs)
    train_masks = np.array(train_imgs_labs)

    return train_images, train_masks

train_images, train_masks = load_images(train_imgs_direc, train_imgs_labs_direc)
print('Loaded images')
new_train_masks = np.empty_like(train_masks)

for i in range(train_masks.shape[0]):
    new_train_masks[i] = cv2.cvtColor(train_masks[i], cv2.COLOR_BGR2RGB)

train_masks = new_train_masks

def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg[np.all(label==background, axis=-1)] = 0
    label_seg[np.all(label==road_flooded, axis=-1)] = 1
    label_seg[np.all(label==road_nonflooded, axis=-1)] = 2

    label_seg = label_seg[:,:,0]

    return label_seg

labels = []
for i in range(train_masks.shape[0]):
    label = rgb_to_2D_label(train_masks[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
print('Labeling done')

labels_cat = to_categorical(labels, num_classes=len(class_colors))
print('One hot encoding performed')
X_train, X_test, y_train, y_test = train_test_split(train_images, labels_cat, test_size=0.2, random_state=42)

print('Split the dataset')

h = X_train.shape[1]
w = X_train.shape[2]
c = X_train.shape[3]

model = UNet(num_classes=len(class_colors), input_size=(h,w,c))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
weights = model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=5, validation_data=(X_test, y_test), shuffle=False)
print(f'Model trained on {4} epochs and a batch size of {16}')
model.save_weights('revamped_two.weights.h5')
y_test_argmax = np.argmax(y_test, axis=3)

test_img_number = random.randint(0, len(X_test))
print('test_img_number index: ', test_img_number)
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]
cv2.imwrite('actual_aerial_img.png', test_img)

plt.figure(figsize=(12,8))
plt.subplot(231)
plt.title('Test Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Test Label')
plt.imshow(ground_truth, cmap='jet')
plt.subplot(233)
plt.title('Predicted Image')
plt.imshow(predicted_img, cmap='jet')
plt.show()
