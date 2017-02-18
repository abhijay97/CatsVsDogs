# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:39:00 2017

@author: abhijay
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Convolution2D(32,3,3,activation='relu'))#convolution
classifier.add(MaxPooling2D(pool_size=(2,2)))#pooling

classifier.add(Convolution2D(32,3,3,input_shape=(256,256,3),activation='relu'))#convolution
classifier.add(MaxPooling2D(pool_size=(2,2)))#pooling

classifier.add(Flatten())#flattening
classifier.add(Dense(output_dim=128,activation = 'relu'))#hidden layer
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))#output layer

classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/home/abhijay/Documents/dataset/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/home/abhijay/Documents/dataset/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=100,
        validation_data=test_set,
        nb_val_samples=2000)
