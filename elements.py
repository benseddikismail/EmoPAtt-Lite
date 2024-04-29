import shutil
import subprocess
import os
import pandas as pd
import numpy as np
import h5py
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight


class CBAMLayer(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.channel_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.channel_max_pool = tf.keras.layers.MaxPooling2D()
        self.channel_dense_1 = tf.keras.layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.channel_dense_2 = tf.keras.layers.Dense(self.channels, activation='sigmoid')
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=(7, 7), padding='same', activation='sigmoid')

    def call(self, inputs):
        channel_avg_pool = self.channel_avg_pool(inputs)
        channel_max_pool = self.channel_max_pool(inputs)
        batch_size = tf.shape(inputs)[0]
        channel_avg_pool = tf.keras.layers.Reshape((1, 1, self.channels))(tf.tile(tf.expand_dims(channel_avg_pool, axis=1), [1, 1, self.channels]))
        channel_max_pool = tf.keras.layers.Reshape((1, 1, self.channels))(tf.tile(tf.expand_dims(channel_max_pool, axis=1), [1, 1, self.channels]))
        channel_avg_pool = self.channel_dense_1(channel_avg_pool)
        channel_max_pool = self.channel_dense_1(channel_max_pool)
        channel_avg_pool = self.channel_dense_2(channel_avg_pool)
        channel_max_pool = self.channel_dense_2(channel_max_pool)
        channel_attention = tf.keras.layers.Add()([channel_avg_pool, channel_max_pool])
        channel_attention = tf.keras.layers.Multiply()([inputs, channel_attention])

        spatial_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(channel_attention)
        spatial_max_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(channel_attention)
        spatial_concat = tf.keras.layers.Concatenate(axis=-1)([spatial_avg_pool, spatial_max_pool])
        spatial_attention = self.spatial_conv(spatial_concat)
        spatial_attention = tf.keras.layers.Multiply()([channel_attention, spatial_attention])

        return spatial_attention


class SpatialTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, interpolation='BILINEAR', **kwargs):
        self.output_size = output_size
        self.interpolation = interpolation
        super(SpatialTransformerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpatialTransformerLayer, self).build(input_shape)

    def call(self, inputs):
        localization_network = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, kernel_size=7, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(10, kernel_size=5, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(6)  
        ], name='localization_network')

        theta = localization_network(inputs)
        theta = tf.keras.layers.Reshape((2, 3))(theta)

        grid = self._meshgrid(self.output_size[0], self.output_size[1])
        output = self._transform(inputs, theta, grid)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.output_size + (input_shape[-1],)

    def _meshgrid(self, height, width):
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)
        return tf.stack([x_t, y_t], axis=-1)

    def _transform(self, image, theta, grid):
        grid = tf.reshape(grid, [-1, self.output_size[0] * self.output_size[1], 2])
        grid = tf.matmul(grid, theta)
        grid = tf.reshape(grid, [-1, self.output_size[0], self.output_size[1], 2])

        # sampling
        output = tf.raw_ops.ImageProjectiveTransformV2(
            images=image,
            transforms=grid,
            output_shape=tf.shape(image)[1:3],
            interpolation=self.interpolation,
            name='transformed'
        )

        return output


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return inputs * tf.expand_dims(tf.expand_dims(x, 1), 1)

