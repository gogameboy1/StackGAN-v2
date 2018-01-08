import tensorflow as tf

import numpy as np

import pdb


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def upsampling_block(input):

    input = tf.reshape(input, [-1, 4, 4, 64*32])

    hidden = tf.contrib.layers.convolution2d_transpose(
        inputs=input,
        num_outputs=256,
        kernel_size=[2, 2],
        stride=[2, 2],
        activation_fn=tf.identity,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    hidden = tf.contrib.layers.convolution2d_transpose(
        inputs=hidden,
        num_outputs=256,
        kernel_size=[4, 4],
        stride=[2, 2],
        activation_fn=tf.identity,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    hidden = tf.contrib.layers.convolution2d_transpose(
        inputs=hidden,
        num_outputs=128,
        kernel_size=[6, 6],
        stride=[2, 2],
        activation_fn=tf.identity,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    output = tf.contrib.layers.convolution2d_transpose(
        inputs=hidden,
        num_outputs=128,
        kernel_size=[8, 8],
        stride=[2, 2],
        activation_fn=tf.tanh,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    return output


# ToDo: how to add the x to CNN output
def residual(input, channel):

    res = input

    hidden = tf.contrib.layers.convolution2d(
        inputs=input,
        num_outputs=128,
        kernel_size=[1, 1],
        stride=1,
        padding='valid',
        activation_fn=tf.identity
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    hidden = tf.contrib.layers.convolution2d(
        inputs=hidden,
        num_outputs=128,
        kernel_size=[4, 4],
        stride=1,
        padding='same',
        activation_fn=tf.identity
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    hidden = tf.contrib.layers.convolution2d(
        inputs=hidden,
        num_outputs=channel,
        kernel_size=[1, 1],
        stride=1,
        padding='valid',
        activation_fn=tf.identity
    )

    hidden = tf.contrib.layers.batch_norm(hidden)
    hidden = leaky_relu(hidden)

    output = hidden + res

    return output


def residual_block_64(input, condition):

    joined_input = tf.concat([input, condition], axis=1)

    hidden = tf.contrib.layers.fully_connected(
        inputs=joined_input,
        num_outputs=64 * 64 * 4 * 32,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )

    hidden = leaky_relu(hidden)

    hidden = tf.reshape(hidden, [-1, 64, 64, 4 * 32])

    hidden = residual(hidden, 128)

    hidden = residual(hidden, 128)

    output = tf.contrib.layers.convolution2d_transpose(
        inputs=hidden,
        num_outputs=64,
        kernel_size=[8, 8],
        stride=[2, 2],
        activation_fn=tf.tanh,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    return output


def residual_block_128(input, condition):

    joined_input = tf.concat([input, condition], axis=1)

    hidden = tf.contrib.layers.fully_connected(
        inputs=joined_input,
        num_outputs=128 * 128 * 2 * 32,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )

    hidden = leaky_relu(hidden)

    hidden = tf.reshape(hidden, [-1, 128, 128, 2 * 32])

    hidden = residual(hidden, 64)

    hidden = residual(hidden, 64)

    output = tf.contrib.layers.convolution2d_transpose(
        inputs=hidden,
        num_outputs=32,
        kernel_size=[8, 8],
        stride=[2, 2],
        activation_fn=tf.tanh,
        weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
    )

    return output


def conv3x3(input):

    conv = tf.contrib.layers.convolution2d(
        inputs=input,
        num_outputs=3,
        kernel_size=[1, 1],
        stride=1,
        padding='valid',
        activation_fn=tf.tanh
    )

    return conv


class Generator(object):

    def __init__(self):
        self.name = 'generator'

    def __call__(self, noise, embed_condition, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):

            input = tf.concat([noise, embed_condition], axis=1)

            embed_input = tf.contrib.layers.fully_connected(
                inputs=input,
                num_outputs=4 * 4 * 64 * 32,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
            )

            g_output_first = upsampling_block(embed_input)

            g_output_for_d_1 = conv3x3(g_output_first)

            g_output_first = tf.contrib.layers.flatten(g_output_first)
            g_output_second = residual_block_64(g_output_first, embed_condition)

            g_output_for_d_2 = conv3x3(g_output_second)

            g_output_second = tf.contrib.layers.flatten(g_output_second)
            g_output_final = residual_block_128(g_output_second, embed_condition)

            g_output_for_d_final = conv3x3(g_output_final)

            return g_output_for_d_1, g_output_for_d_2, g_output_for_d_final

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator_64(object):

    def __init__(self):
        self.name = 'discriminator_0'

    def __call__(self, images, condition, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):

            images = tf.image.resize_images(images, [64, 64])

            hidden = tf.contrib.layers.convolution2d(
                inputs=images,
                num_outputs=512,
                kernel_size=[4, 4],
                stride=4,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.convolution2d(
                inputs=hidden,
                num_outputs=8 * 64,
                kernel_size=[4, 4],
                stride=4,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.flatten(hidden)

            unconditional_loss, conditional_loss = tf.split(hidden,
                                                            num_or_size_splits=2,
                                                            axis=1)

            conditional_loss = tf.concat([conditional_loss, condition], axis=1)

            unconditional_loss = tf.contrib.layers.fully_connected(
                inputs=unconditional_loss,
                num_outputs=128,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            unconditional_loss = leaky_relu(unconditional_loss)

            unconditional_loss = tf.contrib.layers.fully_connected(
                inputs=unconditional_loss,
                num_outputs=1,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=160,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            conditional_loss = leaky_relu(conditional_loss)

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=1,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            return unconditional_loss, conditional_loss

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator_128(object):

    def __init__(self):
        self.name = 'discriminator_1'

    def __call__(self, images, condition, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):

            images = tf.image.resize_images(images, [128, 128])

            hidden = tf.contrib.layers.convolution2d(
                inputs=images,
                num_outputs=512,
                kernel_size=[8, 8],
                stride=4,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.convolution2d(
                inputs=hidden,
                num_outputs=512,
                kernel_size=[5, 5],
                stride=2,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.convolution2d(
                inputs=hidden,
                num_outputs=8 * 64,
                kernel_size=[2, 2],
                stride=4,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.flatten(hidden)

            unconditional_loss, conditional_loss = tf.split(hidden,
                                                            num_or_size_splits=2,
                                                            axis=1)

            conditional_loss = tf.concat([conditional_loss, condition], axis=1)

            unconditional_loss = tf.contrib.layers.fully_connected(
                inputs=unconditional_loss,
                num_outputs=128,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            unconditional_loss = leaky_relu(unconditional_loss)

            unconditional_loss = tf.contrib.layers.fully_connected(
                unconditional_loss,
                num_outputs=1,
                activation_fn=tf.identity
            )

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=160,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            conditional_loss = leaky_relu(conditional_loss)

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=1,
                activation_fn=tf.identity
            )

            return unconditional_loss, conditional_loss

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Discriminator_256(object):

    def __init__(self):
        self.name = 'discriminator_2'

    def __call__(self, images, condition, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):

            hidden = tf.contrib.layers.convolution2d(
                inputs=images,
                num_outputs=512,
                kernel_size=[8, 8],
                stride=4,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.convolution2d(
                inputs=hidden,
                num_outputs=512,
                kernel_size=[6, 6],
                stride=3,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.convolution2d(
                inputs=hidden,
                num_outputs=8 * 64,
                kernel_size=[3, 3],
                stride=5,
                activation_fn=tf.identity
            )

            hidden = leaky_relu(hidden)

            hidden = tf.contrib.layers.flatten(hidden)

            unconditional_loss, conditional_loss = tf.split(hidden,
                                                            num_or_size_splits=2,
                                                            axis=1)

            conditional_loss = tf.concat([conditional_loss, condition], axis=1)

            unconditional_loss = tf.contrib.layers.fully_connected(
                inputs=unconditional_loss,
                num_outputs=128,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            unconditional_loss = leaky_relu(unconditional_loss)

            unconditional_loss = tf.contrib.layers.fully_connected(
                inputs=unconditional_loss,
                num_outputs=1,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=160,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            conditional_loss = leaky_relu(conditional_loss)

            conditional_loss = tf.contrib.layers.fully_connected(
                inputs=conditional_loss,
                num_outputs=1,
                weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            return unconditional_loss, conditional_loss

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]



