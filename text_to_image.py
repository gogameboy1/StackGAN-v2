import tensorflow as tf
import numpy as np

import pdb
import json
import random
import time
import argparse
import scipy.misc
import skimage.io
import skimage.transform

import model


class TextToImage(object):

    def __init__(self, G_net, D_net_0, D_net_1, D_net_2):
        self.g_net = G_net
        self.d_net_0 = D_net_0
        self.d_net_1 = D_net_1
        self.d_net_2 = D_net_2

        self.z_dim = 128
        self.x_dim = 64 * 64 * 3
        self.condition_dim = 2400

        self.images, self.right_texts, self.wrong_texts = self.read_and_decode('train_image.tfrecords')

        self.z = tf.random_uniform(shape=[32, self.z_dim], minval=-1.0, maxval=1.0, seed=1)

        self.embed_condition = tf.contrib.layers.fully_connected(
            inputs=self.right_texts,
            num_outputs=160,
            weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )

        self.embed_wrong_condition = tf.contrib.layers.fully_connected(
            inputs=self.wrong_texts,
            num_outputs=160,
            weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )

        self.d0_input, self.d1_input, self.d2_input = self.g_net(self.z, self.embed_condition)

        self.d0_right_text_loss, self.cond_d0_right_text_loss = self.d_net_0(self.images,
                                                                             self.embed_condition,
                                                                             reuse=False)
        self.d1_right_text_loss, self.cond_d1_right_text_loss = self.d_net_1(self.images,
                                                                             self.embed_condition,
                                                                             reuse=False)
        self.d2_right_text_loss, self.cond_d2_right_text_loss = self.d_net_2(self.images,
                                                                             self.embed_condition,
                                                                             reuse=False)

        self.d0_wrong_text_loss, self.cond_d0_wrong_text_loss = self.d_net_0(self.images, self.embed_wrong_condition)
        self.d1_wrong_text_loss, self.cond_d1_wrong_text_loss = self.d_net_1(self.images, self.embed_wrong_condition)
        self.d2_wrong_text_loss, self.cond_d2_wrong_text_loss = self.d_net_2(self.images, self.embed_wrong_condition)

        self.d0_fake_image_loss, self.cond_d0_fake_image_loss = self.d_net_0(self.d0_input, self.embed_condition)
        self.d1_fake_image_loss, self.cond_d1_fake_image_loss = self.d_net_1(self.d1_input, self.embed_condition)
        self.d2_fake_image_loss, self.cond_d2_fake_image_loss = self.d_net_2(self.d2_input, self.embed_condition)

        self.d0_loss = tf.reduce_mean((self.d0_right_text_loss + self.cond_d0_right_text_loss) / 2) - \
            tf.reduce_mean((self.d0_wrong_text_loss + self.d0_fake_image_loss + self.cond_d0_wrong_text_loss +
                            self.cond_d0_wrong_text_loss) / 4)

        self.d1_loss = tf.reduce_mean((self.d1_right_text_loss + self.cond_d1_right_text_loss) / 2) - \
            tf.reduce_mean((self.d1_wrong_text_loss + self.d1_fake_image_loss + self.cond_d1_wrong_text_loss +
                            self.cond_d1_fake_image_loss) / 4)

        self.d2_loss = tf.reduce_mean((self.d2_right_text_loss + self.cond_d2_right_text_loss) / 2) - \
            tf.reduce_mean((self.d2_wrong_text_loss + self.d2_fake_image_loss + self.cond_d2_wrong_text_loss +
                            self.cond_d2_fake_image_loss) / 4)

        # add improved WGAN penalty to three discriminator
        images_for_d0 = tf.image.resize_images(self.images, [64, 64])
        images_for_d1 = tf.image.resize_images(self.images, [128, 128])

        images_for_d0 = tf.contrib.layers.flatten(images_for_d0)
        images_for_d1 = tf.contrib.layers.flatten(images_for_d1)
        images_for_d2 = tf.contrib.layers.flatten(self.images)

        fake_image_for_d0 = tf.contrib.layers.flatten(self.d0_input)
        fake_image_for_d1 = tf.contrib.layers.flatten(self.d1_input)
        fake_image_for_d2 = tf.contrib.layers.flatten(self.d2_input)

        epsilon = tf.random_uniform([], minval=0.0, maxval=1.0)
        x0_penalty_flat = epsilon * images_for_d0 + (1 - epsilon) * fake_image_for_d0
        x1_penalty_flat = epsilon * images_for_d1 + (1 - epsilon) * fake_image_for_d1
        x2_penalty_flat = epsilon * images_for_d2 + (1 - epsilon) * fake_image_for_d2

        x0_penalty = tf.reshape(x0_penalty_flat, [-1, 64, 64, 3])
        x1_penalty = tf.reshape(x1_penalty_flat, [-1, 128, 128, 3])
        x2_penalty = tf.reshape(x2_penalty_flat, [-1, 256, 256, 3])

        d0_penalty = self.d_net_0(x0_penalty, self.embed_condition)
        d1_penalty = self.d_net_1(x1_penalty, self.embed_condition)
        d2_penalty = self.d_net_2(x2_penalty, self.embed_condition)

        d_d0_penalty = tf.gradients(d0_penalty[0] + d0_penalty[1], x0_penalty_flat)
        d_d0_penalty = tf.sqrt(tf.reduce_sum(tf.square(d_d0_penalty), axis=1))
        d_d0_penalty = tf.reduce_mean(tf.square(d_d0_penalty - 1.0) * 10.0)

        d_d1_penalty = tf.gradients(d1_penalty[0] + d1_penalty[1], x1_penalty_flat)
        d_d1_penalty = tf.sqrt(tf.reduce_sum(tf.square(d_d1_penalty), axis=1))
        d_d1_penalty = tf.reduce_mean(tf.square(d_d1_penalty - 1.0) * 10.0)

        d_d2_penalty = tf.gradients(d2_penalty[0] + d2_penalty[1], x2_penalty_flat)
        d_d2_penalty = tf.sqrt(tf.reduce_sum(tf.square(d_d2_penalty), axis=1))
        d_d2_penalty = tf.reduce_mean(tf.square(d_d2_penalty - 1.0) * 10.0)

        self.d_loss = self.d0_loss + d_d0_penalty + self.d1_loss + d_d1_penalty + self.d2_loss + d_d2_penalty

        self.g_loss = tf.reduce_mean((self.d0_fake_image_loss + self.cond_d0_fake_image_loss + self.d1_fake_image_loss +
                                      self.cond_d1_fake_image_loss + self.d2_fake_image_loss +
                                      self.cond_d2_fake_image_loss) / 6)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=[self.d_net_0.vars, self.d_net_1.vars, self.d_net_2.vars])
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def generate(self):
        pass

    def train(self):

        saver = tf.train.Saver(max_to_keep=1)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start_time = time.time()

        print ("start training, time : " + str(start_time) + " ...")

        for episode in range(0, 5000):

            d_iters = 5

            if episode % 500 == 0 or episode < 25:
                d_iters = 100

            for _ in range(0, d_iters):
                self.sess.run(self.d_adam)

            self.sess.run(self.g_adam)

            if episode % 100 == 0 and episode != 0:
                if episode % 500 == 0:
                    saver.save(self.sess, 'stackGAN_v2_model/model', global_step=episode)

                d_loss = self.sess.run(self.d_loss)
                g_loss = self.sess.run(self.g_loss)

                print ("Episode %s, Time %s, d_loss :%s, g_loss :%s\n" %
                       (str(episode), str(time.time() - start_time), str(np.mean(d_loss)), str(np.mean(g_loss))))

                fake_image = self.sess.run(self.d2_input)
                fake_image = np.reshape(fake_image, (fake_image.shape[0], 256, 256, 3))
                try:
                    for index, pic in enumerate(fake_image[0: 5]):
                        scipy.misc.imsave("stackGAN_v2_image/sample_" + str(episode) + "_" + str(index) + ".jpg",
                                              pic)
                except Error:
                    continue

        coord.request_stop()
        coord.join(threads)

        print ("Done")

    @staticmethod
    def read_and_decode(tfrecord_path):

        filename_queue = tf.train.string_input_producer([tfrecord_path])

        reader = tf.TFRecordReader()

        _, serialized_data = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_data,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string),
                'right_text': tf.FixedLenFeature([], tf.string),
                'wrong_text': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        right_text = tf.decode_raw(features['right_text'], tf.float32)
        wrong_text = tf.decode_raw(features['wrong_text'], tf.float32)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image = tf.reshape(image, [height, width, 3])

        resized_image = tf.image.resize_images(image, [256, 256])

        images, right_texts, wrong_texts = tf.train.shuffle_batch(
            [resized_image, right_text, wrong_text],
            batch_size=32,
            num_threads=2,
            capacity=80,
            min_after_dequeue=10,
            shapes=[[256, 256, 3], [2400], [2400]]
        )

        return images, right_texts, wrong_texts


if __name__ == '__main__':

    np.random.seed(1)

    parser = argparse.ArgumentParser('')
    parser.add_argument('--train', help='train', action='store_true')
    parser.add_argument('--generate', help='generate picture', action='store_true')
    parser.add_argument('--testing_file', help='testing conditions file')
    args = parser.parse_args()

    G_net = model.Generator()
    D_net_0 = model.Discriminator_64()
    D_net_1 = model.Discriminator_128()
    D_net_2 = model.Discriminator_256()

    text2image = TextToImage(G_net=G_net,
                             D_net_0=D_net_0,
                             D_net_1=D_net_1,
                             D_net_2=D_net_2)

    if args.train:
        text2image.train()


