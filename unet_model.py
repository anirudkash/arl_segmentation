import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras import backend as K

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def UNet(num_classes, input_size=(256,256,10)):

    inputs = Input(input_size)

    conv1 = Conv2D(16,(3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, (3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    
    conv2 = Conv2D(32, (3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, (3,3), activation='relu',
                   kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Conv2D(64, (3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3,3), activation='relu',
                   kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(128, (3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3,3), activation='relu',
                   kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    
    conv5 = Conv2D(256, (3,3), activation='relu', 
                   kernel_initializer='he_normal', padding='same')(pool4)
    conv5= Dropout(0.3)(conv5)
    conv5 = Conv2D(256, (3,3), activation='relu',
                   kernel_initializer='he_normal', padding='same')(conv5)
    
    # Upsampling Stage
    up6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

    up8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(up8) 
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

    up9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)   
    
    outputs = Conv2D(num_classes, (1,1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model