# -*- coding: utf-8 -*-
from keras.models import Model, Sequential, Input

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector

from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D


def model_3d(nb_category, shape):
    model = Sequential()
    model.add(Reshape((shape[0], shape[1], shape[2], 1), input_shape=shape))
    model.add(Conv3D(16, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Conv3D(32, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Conv3D(64, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(128, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(256, (3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_category))
    # model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))
    # model = Sequential()
    # model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
    #     64, 64, 32, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    # model.add(Activation('softmax'))
    # model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    # model.add(Activation('softmax'))
    # model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512, activation='sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_category, activation='sigmoid'))
    model.summary()
    return model
