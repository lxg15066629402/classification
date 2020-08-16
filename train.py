# -*- coding: utf-8 -*-

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import json
import keras_models
import keras_models_
import keras.backend as K
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.losses import categorical_crossentropy, binary_crossentropy
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

np.random.seed(42)
CATEGORY = 2
BATCH_SIZE = 8
EPOCHS = 1000
d = 229  # type: int
input_shape = (64, 64, 16)
session = 'MG'
fold = 4
models = ["vgg16", "res50", "inception3", "inception_res_v2", "dense", 'xcep']


def display_img(image):
    plt.imshow(image.copy(), cmap='gray')
    plt.show()


def make_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def make_model_dir(func):
    def wrapper(*args, **kwargs):
        global session, model_path
        if session != '':
            model_path = './keras-model/' + session
            make_dir(model_path)
        return func(*args, **kwargs)
    return wrapper


def train(train_x, train_y, valid_x, valid_y):
    def preprocess(x):
        x /= 127.5
        x -= 1.
        return x

    gen = ImageDataGenerator(
        rotation_range=10.,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        preprocessing_function=preprocess
    )

    valid_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess
    )

    train_gen = gen.flow(train_x,
                         y=train_y,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         sample_weight=None,
                         seed=42)

    valid_gen = valid_data_gen.flow(valid_x,
                                    y=valid_y,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    sample_weight=None,
                                    seed=42)

    model = keras_models_.model_3d(2, input_shape)
    # model = keras_models.model_res_3d(2, input_shape)

    model_checkpoint = ModelCheckpoint(filepath='model/ResUnet.hdf5', save_best_only=True, verbose=1)

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=5e-6), metrics=['accuracy'])
    # categorical_crossentropy
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_x) // BATCH_SIZE + 1,
        epochs=5000,
        validation_data=valid_gen,
        validation_steps=len(valid_x) // BATCH_SIZE + 1,
        callbacks=[model_checkpoint, TensorBoard(log_dir="model/logs")])

    log_dir = 'model/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)


if __name__ == '__main__':

    print("loading dataset done\n")
    # load npy data
    data = np.load("/media/03/data.npy").astype(np.float32)
    labels = np.load("/media/03/label1.npy").astype(np.uint8)

    shp = labels.shape[0]
    Y_labels = labels.reshape(-1)
    Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
    Y_labels = Y_labels.reshape(shp, 2)

    train_x, valid_x, train_y, valid_y = train_test_split(data, Y_labels, test_size=0.2, random_state=0)

    train(train_x, train_y, valid_x, valid_y)
    print("train finished")

