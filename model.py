from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Cropping2D, Lambda, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping
import csv
import matplotlib.pyplot as plt
import numpy as np
from random import random


# Input shape
input_shape = (160, 320, 3)
# Cropped image shape
crop_vertical = (70, 25)
cropped_shape = (3, input_shape[0]-crop_vertical[0], input_shape[1]-crop_vertical[1])

epochs = 5
batch_size = 32

data_dir = 'data/'
images, angles = [], []
training_path = ['recovery-clockwise',
                 '3-laps-clockwise-track1',
                 '3-laps-counter-clockwise-track1']

# Read images and angles
for path in training_path:
    with open(data_dir+path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            steering_center = float(line[3])

            # Because the zero values are dominant, 
            # we randomly drop (85% prob) some to get a Gaussian-like distribution
            if steering_center == 0 and random() <=0.85:
                continue

            # create adjusted steering measurements for the side camera images
            correction = 0.25 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            if path == 'udacity-example':
                IMG_dir = data_dir+path+'/'
            else:
                IMG_dir = data_dir+path+'/IMG/'
            img_center = np.array(plt.imread(IMG_dir+line[0].split('\\')[-1]))
            img_left = np.array(plt.imread(IMG_dir+line[1][1:].split('\\')[-1]))
            img_right = np.array(plt.imread(IMG_dir+line[2][1:].split('\\')[-1]))

            # add images and angles to data set
            images.extend([img_center, img_left, img_right])
            angles.extend([steering_center, steering_left, steering_right])


# Flip augmentation
augmented_images, augmented_angles = [], []
for image, angle in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(np.fliplr(image))
    augmented_angles.append(-angle)


# config Keras generators
# for training data
train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

# for valiation data
valid_datagen = ImageDataGenerator()

# Define generator method
def generator(X, y, type):
    '''
    A helper method create data generator
    Inputs:
        X: images
        y: angles
        type: string, train or valid
    Outputs:
        datagen: a Keras data generator instance
    '''
    # create generator
    X, y = np.array(X), np.array(y)
    if type=='train':
        datagen = train_datagen.flow(X, y, batch_size=batch_size)
    else:
        datagen = valid_datagen.flow(X, y, batch_size=batch_size)

    return datagen


# compile and train the model using the generator function
X_train, X_valid, y_train, y_valid = train_test_split(augmented_images, augmented_angles, test_size=0.2)
train_generator = generator(X_train, y_train, 'train')
validation_generator = generator(X_valid, y_valid, 'valid')


# Create the nVidia model
model = Sequential()
# Crop image to only see the road
model.add(Lambda(lambda x: x/255.-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=(crop_vertical, (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))


# Train the model
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit_generator(train_generator, steps_per_epoch= len(X_train)//batch_size, \
                    validation_data=validation_generator, \
                    validation_steps=len(X_valid)//batch_size, \
                    epochs=epochs, verbose=1, callbacks=[early_stopping])

# Save the model
model.save('model.h5')