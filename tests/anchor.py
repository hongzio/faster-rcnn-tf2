import unittest
import numpy as np
import tensorflow as tf

from frcnn.util.anchor import make_anchors, broadcast_iou, transform
from frcnn.config.config import Config


class TestAnchor(unittest.TestCase):
    def test_make_anchors(self):
        cfg = Config()
        anchors = make_anchors(cfg)
        result = np.array_equal(anchors.numpy(), np.array([[40., 40.],
                                                           [40., 60.],
                                                           [60., 60.],
                                                           [60., 90.]]))
        self.assertTrue(result)

    def test_broadcast_iou(self):
        box1 = tf.constant([[0, 0, 10, 10]], dtype=tf.float32)
        box2 = tf.constant([[[0, 9, 10, 19], [0, 10, 10, 15]]], dtype=tf.float32)
        iou = broadcast_iou(box1, box2)
        expected = np.array([[1 / 19., 0.]], dtype=np.float32)
        result = np.array_equal(iou.numpy(), expected)
        self.assertTrue(result)

    def test_transform(self):
        bboxes = tf.convert_to_tensor([[0.03049111, 0.18494931, 0.96302897, 0.9435849, 0],
                                       [0.35938117, 0.01586703, 0.6069674, 0.17582396, 56],
                                       [0.48252046, 0.09158827, 0.6403017, 0.26967454, 67]])

        anchors = make_anchors(Config('config/transform.yml'))
        input_shape = np.array((416, 416))
        output_sizes = [input_shape / 4, input_shape / 8, input_shape / 16, input_shape / 32]
        y, bboxes = transform(bboxes, anchors, output_sizes, 0.3, 0.7)


if __name__ == '__main__':
    unittest.main()
