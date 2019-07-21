from frcnn.backbone.vgg import VGG
from frcnn.dataset.dataset import fake_dataset
from frcnn.loss import rpn_loss
from frcnn.network.frcnn import FasterRCNNModel
from frcnn.network.rpn import RPN
from frcnn.util.anchor import make_anchors, transform

import tensorflow as tf


class FasterRCNN:
    def __init__(self, config):
        self.config = config
        self.anchors = make_anchors(config)
        self.model = FasterRCNNModel(self.anchors, VGG,
                                     self.config['train']['batch_size'],
                                     self.config['data']['num_classes'],
                                     self.config['train']['max_num_rois'],
                                     self.config['train']['roi_size'],
                                     self.config['train']['roi_overlap_threshold'],
                                     )
        self.dataset = None

    def init_train_context(self):
        dataset = fake_dataset()
        dataset = dataset.map(lambda image, gt_boxes: (tf.image.resize(image, (416, 416)), gt_boxes))
        dataset = dataset.map(lambda image, gt_boxes: (image, transform(gt_boxes,
                                                                        self.anchors,
                                                                        self.model.rpn.backbone.calc_output_size(
                                                                            tf.shape(image)),
                                                                        self.config['train']['min_iou'],
                                                                        self.config['train']['max_iou'],
                                                                        self.config['train']['max_num_gt_boxes'])))
        self.dataset = dataset.batch(self.config['train']['batch_size'])

    @tf.function
    def _train_step(self, x, rpn_y, gt_boxes):
        self.model(x, rpn_y, gt_boxes)
        return tf.reduce_sum(self.model.losses)
        # with tf.GradientTape(persistent=True) as tape:
        #     losses = self.model.losses
        # gradient = tape.gradient(losses, self.model.trainable_variables)
        # del tape
        # self.rpn_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        # return tf.reduce_sum(losses)

    def train(self):
        self.init_train_context()
        for epoch in range(10):
            for step, (x, (rpn_y, gt_boxes)) in enumerate(self.dataset):
                loss = self._train_step(x, rpn_y, gt_boxes)
                print(loss.numpy())
