import tensorflow as tf
import os

def fake_dataset():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    girl = tf.image.decode_png(open(os.path.join(src_dir, 'girl.png'), 'rb').read(), channels=3)
    horse = tf.image.decode_jpeg(open(os.path.join(src_dir, 'horse.jpg'), 'rb').read(), channels=3)

    # y1, x1, y2, x2, cls
    girl_bb = tf.convert_to_tensor([[0.03049111, 0.18494931, 0.96302897, 0.9435849, 0],
                                       [0.35938117, 0.01586703, 0.6069674, 0.17582396, 56],
                                       [0.48252046, 0.09158827, 0.6403017, 0.26967454, 67]])

    horse_bb = tf.convert_to_tensor([
        [0.19683257918552036, 0.106, 0.9502262443438914, 0.942, 13],
        [0.09954751131221719, 0.316, 0.3778280542986425, 0.578, 15]
    ])

    def _data_generator():
        yield girl, girl_bb
        yield horse, horse_bb

    dataset = tf.data.Dataset.from_generator(_data_generator, (tf.int32, tf.float32), ((None, None, 3), (None, 5)))
    return dataset


def voc_dataset():
    pass