from frcnn.backbone.vgg import VGG
from frcnn.dataset.dataset import fake_dataset
from frcnn.loss import rpn_loss
from frcnn.network.rpn import RPN
from frcnn.util.anchor import make_anchors, transform

import tensorflow as tf

class FasterRCNN:
    def __init__(self, config):
        self.config = config
        self.anchors = make_anchors(config)
        self.rpn_model = RPN(len(self.anchors), VGG)
        self.rpn_optimizer = tf.optimizers.Adam(lr=1e-4)
        dataset = fake_dataset()
        dataset = dataset.map(lambda image, bboxes: (tf.image.resize(image, (416, 416)), bboxes))
        dataset = dataset.map(lambda image, bboxes: (image, transform(bboxes,
                                                                   self.anchors,
                                                                   self.rpn_model.backbone.calc_output_size(tf.shape(image)),
                                                                   config['train']['min_iou'],
                                                                   config['train']['max_iou'],
                                                                   config['train']['max_num_boxes'])))
        self.dataset = dataset.batch(2)
        self.rpn_loss = rpn_loss

    def train(self):
        # @tf.function
        def train_step(x, ys):
            losses = []
            with tf.GradientTape() as tape:
                preds = self.rpn_model(x)
                for y, pred in zip(ys, preds):
                    losses.append(rpn_loss(y, pred))
            loss = tf.reduce_sum(losses)
            gradient = tape.gradient(losses, self.rpn_model.trainable_variables)
            self.rpn_optimizer.apply_gradients(zip(gradient, self.rpn_model.trainable_variables))
            return loss
        for epoch in range(100):
            for step, (x, ys) in enumerate(self.dataset):
                loss = train_step(x, ys)
                print(loss.numpy())