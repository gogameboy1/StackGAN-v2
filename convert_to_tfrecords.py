import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np

import os
import json
import pdb
import random
import pickle

from skip_thoughts import configuration
from skip_thoughts import encoder_manager


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


VOCAB_FILE = "skip_thoughts_model/pretrained/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "skip_thoughts_model/pretrained/skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "skip_thoughts_model/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"


print ("loading model ...")
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)


data_name, name_to_annotations, _ = pickle.load(open("data/train_info.pkl", "rb"))


tfrecords_filename = 'train_image.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)


img_dir = 'data/train2014/'

print ("converting image and vector ...")
for name in data_name:
    if not os.path.exists(os.path.join(img_dir, 'COCO_train2014_' + str(name))):
        continue
    else:
        img_path = os.path.join(img_dir,  'COCO_train2014_' + str(name))
        img = np.array(skimage.io.imread(img_path))

        caption = name_to_annotations[name][0]
        wrong_caption = name_to_annotations[random.choice(data_name)][0]

        caption_vec = encoder.encode([caption])[0]
        wrong_caption_vec = encoder.encode([wrong_caption])[0]

        height = img.shape[0]
        width = img.shape[1]

        img = img.tostring()
        caption_vec = caption_vec.tostring()
        wrong_caption_vec = wrong_caption_vec.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'right_text': _bytes_feature(caption_vec),
            'wrong_text': _bytes_feature(wrong_caption_vec)
        }))

        writer.write(example.SerializeToString())

writer.close()

print ("Done")
