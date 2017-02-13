import os
import csv
from numpy.random import shuffle

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def get_model():
    model = Sequential()
    # normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    # remove top and bottom portion of image - the sky and car front - 
    # is not useful for drifing the car
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.1))

    model.add(Dense(256))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def load_dataset():
    """
    We load all image locations and their corresponding steering angles.
    For left image we add 0.2 and for right image we subtract 0.2 from the 
    steering angle to compensate. We shuffle the list at the end.
    """
    osamples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            osamples.append(line)

    #print(len(osamples));

    correction = 0.2
    samples=[]
    for sample in osamples[1:]:
        measurement = float(sample[3])
        samples.append([sample[0], measurement])
        samples.append([sample[1], measurement + correction]) #left image
        samples.append([sample[2], measurement - correction]) #right image

    #print(len(samples))
    #print(samples[0])
    #print(samples[4])
    #print(samples[8])

    shuffle(samples)
    return samples

def generator(data,batch_size=32):
    """
    Batch generator creates a generator that sends batch_size amount of images 
    to the caller in each batch. We flip each image to make the dataset bigger.
    
    """
    
    batch_size = int(batch_size /2)
    num_samples = len(data)
    while 1: # Used as a reference pointer so code always loops back around
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]
            #print(offset)
            
            images = []
            angles = []
            #print(batch_samples)
            #Add original image and flipped image with steering angle reversed
            for batch_sample in batch_samples:
                angle = float(batch_sample[1])
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                #print(name)
                image = cv2.imread(name)
                flip_image = cv2.flip(image,1)
                images.append(image)
                angles.append(angle)
                images.append(flip_image)
                angles.append(angle * -1.0)
                #print(center_image.shape)
                #print(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Load data
samples = load_dataset()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model = get_model()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=20)

#Save the model
model.save("model.h5")


