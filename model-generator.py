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
from sklearn.utils import shuffle


# Input shape
input_shape = (160, 320, 3)
# Cropped image shape
crop_vertical = (70, 25)
cropped_shape = (3, input_shape[0]-crop_vertical[0], input_shape[1]-crop_vertical[1])

epochs = 10
batch_size = 32

data_dir = 'data/'
training_path = ['recovery',
                 'recovery-counter-clockwise',
                 '3-laps-clockwise-track1',
                 '3-laps-counter-clockwise-track1',
                 '1-lap-clockwise-track2',
                 '1-lap-counter-clockwise-track2']

# Read csv logs
samples = []
for path in training_path:
    with open(data_dir+path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

# Split the data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define the generator
def generator(samples):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:

                steering_center = float(batch_sample[3])

                # Because the zero values are dominant, 
                # we randomly drop (85% prob) some to get a Gaussian-like distribution
                if steering_center == 0 and random() <= 0.85:
                    continue

                # create adjusted steering measurements for the side camera images
                correction = 0.25 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read images from center, left and right cameras
                sample_image = []
                for i in range(3):
                    # Deal with Windows/Mac differences
                    if batch_sample[i].startswith('/'):
                        split_char = '/'
                    else:
                        split_char = '\\'
                    path_splits = batch_sample[i].split(split_char)
                    sample_image.append(np.array(plt.imread(split_char.join(path_splits[-4:]))))

                # add images and angles to data set
                images.extend(sample_image)
                angles.extend([steering_center, steering_left, steering_right])

                # flip to augment the data
                img_flipped = [np.fliplr(sample_image[0]), np.fliplr(sample_image[1]), np.fliplr(sample_image[2])]
                steering_flipped = [-steering_center, -steering_left, -steering_right]
                images.extend(img_flipped)
                angles.extend(steering_flipped)

            X = np.array(images)
            y = np.array(angles)

            yield shuffle(X, y)


# create the generators
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


# Define the nVidia model
model = Sequential()
# Crop image to only see the road
model.add(Lambda(lambda x: x/255.-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=(crop_vertical, (0,0))))
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
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)
history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size, \
                    validation_data=validation_generator, \
                    validation_steps=len(validation_samples)//batch_size, \
                    epochs=epochs, verbose=1, callbacks=[early_stopping])

# Save the model
model.save('model-gen.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.jpg', bbox_inches='tight')


