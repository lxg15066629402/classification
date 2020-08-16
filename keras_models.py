# coding: utf-8

"""
 keras 库实现分类模型
"""
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet169
from keras.applications.xception import Xception
from keras.models import Model, Sequential, Input
from keras.layers import BatchNormalization
from keras.layers import multiply, add, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, ZeroPadding3D
from keras import utils as keras_utils
from keras.utils import plot_model
import keras.backend as K
import h5py
import loading_weights

# 加载预训练模型
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


# 残差_50模块
def basic_res_50(shape):
    bn_axis = 3

    def identity_block(input_tensor, kernel_size, filters, stage, block):

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

    img_input = Input(shape=shape)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x, img_input


def model_res_5(nb, shape, top=True, name='Res50'):

    x, img_input = basic_res_50(shape=shape)

    x = GlobalAveragePooling2D()(x)

    model = Model(img_input, x, name=name)

    weights_path = keras_utils.get_file(
        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)

    if not top:
        return model
    has_top = Dense(nb, activation='softmax', name='fullconnected')(model.output)
    m = Model(model.input, has_top)
    return m


def res_50_3d(shape):
    bn_axis = 4

    def identity_block(input_tensor, kernel_size, filters, stage, block):

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv3D(filters1, (1, 1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters2, kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters3, (1, 1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2, 2)):

        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv3D(filters1, (1, 1, 1), strides=strides,
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters2, kernel_size, padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv3D(filters3, (1, 1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv3D(filters3, (1, 1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

    img_input = Input(shape=shape)
    x = ZeroPadding3D(padding=(3, 3, 3), name='conv1_pad')(Reshape((shape[0], shape[1], shape[2], 1))(img_input))
    x = Conv3D(64, (7, 7, 7),
               strides=(2, 2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1')(x)

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding3D(padding=(1, 1, 1), name='pool1_pad')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling3D()(x)

    model = Model(img_input, x, name='res_3d')

    # model.summary()
    # plot_model(model)

    return model


def model_res_3d(nb, shape, top=True, name='Res50-3d'):

    model = res_50_3d(shape=shape)
    hs = Dropout(0.5)(model.output)
    has_top = Dense(nb, activation='sigmoid', name='fullconnected')(hs)
    m = Model(model.input, has_top)

    # m.summary()
    # plot_model(model)

    return m


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

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_category))
    model.add(Activation('softmax'))
    model.summary()
    return model


def model_2d(nb_category, shape):

    def check_channel(i):
        return i[..., 0]

    inputs_liver = Input(shape=shape)
    inputs_patch = Input(shape=shape)
    inputs_clinic = Input(shape=(2,))

    def get_model(inputs):
        x = Lambda(check_channel)(inputs)
        x = Reshape((shape[0], shape[1], 1))(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(1024, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        return x

    m1 = get_model(inputs_liver)
    deep1 = Dense(512)(m1)
    deep1 = Dense(nb_category)(deep1)
    deep1 = Activation('softmax', name='liver')(deep1)

    m2 = get_model(inputs_patch)
    deep2 = Dense(512)(m2)
    deep2 = Dense(nb_category)(deep2)
    deep2 = Activation('softmax', name='patch')(deep2)

    mix = concatenate([m1, m2])
    mix = Activation('relu')(mix)
    # mix = Dense(512)(mix)
    # mix = Activation('relu')(mix)
    mix = Dense(nb_category)(mix)
    mix = Activation('softmax', name='mix')(mix)

    model = Model([inputs_liver, inputs_patch, inputs_clinic], [deep1, deep2, mix])
    model.summary()
    plot_model(model)
    return model


def se_block(inputs, ratio=16):
    init = inputs
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def model_se(nb_category, shape):

    inputs_liver = Input(shape=shape)
    inputs_patch = Input(shape=shape)
    inputs_clinic = Input(shape=(2,))

    def extract_by_index(i, j):
        return i[..., j]

    inputs_patches = [Reshape((shape[0], shape[1], 1))(Lambda(extract_by_index, arguments={'j': index})(inputs_patch))
                      for index in range(shape[-1])]

    def cnn_block(inputs, nb, k=(3, 3)):
        x = Conv2D(nb, k, padding='same')(inputs)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

    def get_model_reshape(inputs):
        x = Lambda(extract_by_index, arguments={'j': 0})(inputs)
        x = Reshape((shape[0], shape[1], 1))(x)
        x = cnn_block(x, 16)
        x = cnn_block(x, 32)
        x = cnn_block(x, 128)
        x = cnn_block(x, 256)
        x = cnn_block(x, 512)
        x = GlobalAveragePooling2D()(x)
        return x

    def get_model_se(inputs=Input((shape[0], shape[1], 1))):
        x = cnn_block(inputs, 16)
        x = cnn_block(x, 32)
        x = cnn_block(x, 128)
        x = cnn_block(x, 256)
        x = cnn_block(x, 512)
        pool = GlobalAveragePooling2D()(x)
        return Model(inputs, [x, pool], name='patch_model')

    liver_model = get_model_reshape(inputs_liver)
    deep1 = Dense(512)(liver_model)
    deep1 = Dense(nb_category)(deep1)
    deep1 = Activation('softmax', name='liver')(deep1)

    patch_model = get_model_se()
    se_outs = [patch_model(i) for i in inputs_patches]

    se_outs_pool = [i[1] for i in se_outs]
    se_outs_1_x_1 = [Conv2D(1, (1, 1))(i[0]) for i in se_outs]
    se_outs_1_x_1 = concatenate([Flatten()(i) for i in se_outs_1_x_1])
    se_weight = Dense(16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_outs_1_x_1)
    se_weight = Dense(len(se_outs), activation='softmax', kernel_initializer='he_normal', use_bias=False)(se_weight)

    patch_end_input = Input((512,))
    patch_end = Dense(512)(patch_end_input)
    patch_end = Dense(nb_category)(patch_end)
    patch_end = Activation('softmax', name='deep2')(patch_end)
    patch_end = Model(patch_end_input, patch_end, name='patch_end')

    patch_activated = [patch_end(i) for i in se_outs_pool]
    patch_weighted = Lambda(lambda x: K.stack(x, axis=-1), name='stack_patch')(patch_activated)
    patch_weighted = multiply([patch_weighted, se_weight])
    patch_weighted = Lambda(lambda x: K.mean(x, axis=-1))(patch_weighted)
    patch_weighted = Activation('softmax', name='patch')(patch_weighted)

    patch_mix = Lambda(lambda x: K.stack(x, axis=-1))(se_outs_pool)
    patch_mix = multiply([patch_mix, se_weight])
    patch_mix = Lambda(lambda x: K.mean(x, axis=-1))(patch_mix)
    mix = concatenate([liver_model, patch_mix])
    mix = Dense(nb_category)(mix)
    mix = Activation('softmax', name='mix')(mix)

    model = Model([inputs_liver, inputs_patch, inputs_clinic], [deep1, patch_weighted, mix])
    model.summary()
    plot_model(model)
    return model


def model_se_res(nb, shape):

    inputs_liver = Input(shape=shape, name='Liver')
    inputs_patch = Input(shape=shape, name='Patch')
    inputs_clinic = Input(shape=(2,), name='Clinic')
    single_shape = (shape[0], shape[1], 1)

    def extract_by_index(i, j):
        return i[..., j]

    def get_3_layer(index, inp):
        single_layer = Lambda(extract_by_index, arguments={'j': index}, input_shape=shape)(inp)
        single_layer = Reshape(single_shape)(single_layer)
        single_layer = Lambda(lambda x: K.concatenate([x, x, x]))(single_layer)
        return single_layer

    liver_model = model_res_5(nb=nb, shape=(shape[0], shape[1], 3), top=False)(get_3_layer(0, inputs_liver))
    deep1 = Dense(512)(liver_model)
    deep1 = Dense(nb)(deep1)
    deep1 = Activation('softmax', name='liver')(deep1)

    patch_model = model_res_5(nb=nb, shape=(shape[0], shape[1], 3), top=False, name='PPP')(get_3_layer(0, inputs_patch))
    deep2 = Dense(512)(patch_model)
    deep2 = Dense(nb)(deep2)
    deep2 = Activation('softmax', name='patch')(deep2)

    mix = concatenate([liver_model, patch_model])
    mix = Dense(nb)(mix)
    mix = Activation('softmax', name='mix')(mix)

    model = Model([inputs_liver, inputs_patch, inputs_clinic], [deep1, deep2, mix])
    model.summary()
    plot_model(model)
    return model


def model_5_inputs(nb_category, shape):

    img_input = Input(shape=shape)
    single_shape = (shape[0], shape[1], 1)

    def get_slice(x, index):
        return x[..., index]

    def get_single_layer(index):
        single_layer = Lambda(get_slice, arguments={'index': index}, input_shape=shape)(img_input)
        single_layer = Reshape(single_shape)(single_layer)
        return single_layer

    def get_3_layer(index):
        single_layer = Lambda(get_slice, arguments={'index': index}, input_shape=shape)(img_input)
        single_layer = Reshape(single_shape)(single_layer)
        single_layer = Lambda(lambda x: K.concatenate([x, x, x]))(single_layer)
        return single_layer

    model = model_res_5(nb=nb_category, shape=(shape[0], shape[1], 3), top=False)
    inputs = [model(get_3_layer(i)) for i in range(shape[2])]
    x = Lambda(lambda x: K.sum(x, axis=0))(inputs)

    # x = Activation('relu')(concatenated)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(nb_category)(x)
    x = Activation('softmax')(x)
    model = Model(img_input, x)
    model.summary()
    plot_model(model)
    return model


def model_res_1(nb, shape):
    inputs_liver = Input(shape=shape, name='Liver')
    inputs_patch = Input(shape=shape, name='Patch')
    inputs_clinic = Input(shape=(2,), name='Clinic')

    single_shape = (shape[0], shape[1], 1)

    def extract_by_index(i, j):
        return i[..., j]

    def get_3_layer(index, inp):
        single_layer = Lambda(extract_by_index, arguments={'j': index}, input_shape=shape)(inp)
        single_layer = Reshape(single_shape)(single_layer)
        single_layer = Lambda(lambda x: K.concatenate([x, x, x]))(single_layer)
        return single_layer

    x = model_res_5(nb=nb, shape=(shape[0], shape[1], 3), top=False)
    x = x(get_3_layer(0, inputs_patch))
    x = Dense(nb)(x)
    x = Activation('softmax')(x)

    m = Model([inputs_liver, inputs_patch, inputs_clinic], x)
    m.summary()
    plot_model(m)
    return m


def model_res50(nb_category, shape):
    m = ResNet50(include_top=False, input_shape=shape, pooling='avg')
    x = Dense(nb_category, activation="softmax", name='predictions')(m.output)
    m = Model(m.input, x)
    m.summary()
    return m


def model_vgg16(nb_category, w, h):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(w, h, 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.6)(x)
    predictions = Dense(nb_category, activation='softmax', name='predictions')(x)
    finetuned_model = Model(input=base_model.input, output=predictions)
    return finetuned_model


def model_inceptionv3(nb_category, w, h):
    m = InceptionV3(include_top=False, input_shape=(w, h, 3))
    # for layer in m.layers:
    #     layer.trainable = False
    x = m.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(nb_category, activation="softmax", name='predictions')(x)
    finetuned_model = Model(m.input, x)
    return finetuned_model


def model_inceptionv_resnetv2(nb_category, w, h):
    m = InceptionResNetV2(include_top=False, input_shape=(w, h, 3), pooling='avg')
    # for layer in m.layers:
    #     layer.trainable = True
    x = Dense(nb_category, activation="softmax", name='predictions')(m.output)
    finetuned_model = Model(m.input, x)
    return finetuned_model


def model_dense(nb_category, w, h):

    m = DenseNet169(include_top=False, input_shape=(w, h, 3), pooling='avg')
    # for layer in m.layers:
    #     layer.trainable = False
    x = Dense(nb_category, activation="softmax", name='predictions')(m.output)
    finetuned_model = Model(m.input, x)
    return finetuned_model


def model_xcep(nb_category, w, h):

    m = Xception(include_top=False, input_shape=(w, h, 3), pooling='avg')
    # for layer in m.layers:
    #     layer.trainable = False
    x = Dense(nb_category, activation="softmax", name='predictions')(m.output)
    finetuned_model = Model(m.input, x)
    return finetuned_model
