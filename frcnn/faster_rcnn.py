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
        self.rpn_loss = None
        self.dataset = None
        self.rpn_optimizer = None

    def init_train_context(self):
        self.rpn_optimizer = tf.optimizers.Adam(lr=1e-4)
        self.rpn_loss = rpn_loss
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

    def train(self):
        self.init_train_context()

        # @tf.function
        def train_step(x, rpn_y, gt_boxes):
            losses = []
            with tf.GradientTape() as tape:
                preds = self.model(x, rpn_y, gt_boxes)
                for y, pred in zip(rpn_y, preds):
                    losses.append(rpn_loss(y, pred))
            gradient = tape.gradient(losses, self.model.trainable_variables)
            self.rpn_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
            return tf.reduce_sum(losses)

        for epoch in range(100):
            for step, (x, (rpn_y, gt_boxes)) in enumerate(self.dataset):
                loss = train_step(x, rpn_y, gt_boxes)
                print(loss.numpy())
