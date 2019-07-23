import itertools

from frcnn.backbone.vgg import VGG
from frcnn.dataset.dataset import fake_dataset
from frcnn.loss import rpn_loss, roi_loss
from frcnn.network.classifier import RoiClassifier
from frcnn.network.rpn import RPN
from frcnn.util.anchor import make_anchors, transform, broadcast_iou, make_anchor_coords

import tensorflow as tf
import os
import numpy as np

class Saver(tf.keras.models.Model):
    def __init__(self, rpn, classifier, **kwargs):
        super().__init__(**kwargs)
        self.rpn = rpn
        self.classifier = classifier

class FasterRCNN:
    def __init__(self, config):
        self.config = config
        self.anchors = make_anchors(config)

        self.rpn = RPN(len(self.anchors), VGG)
        self.classifier = RoiClassifier(self.config['data']['num_classes'])
        self.saver = Saver(self.rpn, self.classifier)


        self.dataset = None
        self.train_loss = None
        self.optimizer = None
        self.ckpt = None
        self.ckpt_manager = None

        self._roi_size = self.config['train']['roi_size']


    def _init_train_context(self):
        dataset = fake_dataset()
        dataset = dataset.map(lambda image, gt_boxes: (tf.image.resize(image, (416, 416)), gt_boxes))
        dataset = dataset.map(lambda image, gt_boxes: (image, transform(gt_boxes,
                                                                        self.anchors,
                                                                        self.rpn.backbone.calc_output_size(tf.shape(image)),
                                                                        self.config['train']['min_iou'],
                                                                        self.config['train']['max_iou'],
                                                                        self.config['train']['max_num_gt_boxes'])))
        self.dataset = dataset.batch(self.config['train']['rpn_batch_size'])

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer = tf.keras.optimizers.Adam(float(self.config['train']['lr']))

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, rpn=self.rpn, classifier=self.classifier)
        if not os.path.exists(self.config['train']['checkpoint_dir']):
            os.mkdir(self.config['train']['checkpoint_dir'])
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       self.config['train']['checkpoint_dir'],
                                                       max_to_keep=3)
        # self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        # self.rpn.build(input_shape=(None, 416, 416, 3))
        # self.classifier.build(input_shape=(None, self._roi_size, self._roi_size, 512))
        # self.saver.load_weights(self.config['train']['model_file'])

    def _calc_roi_target(self, pred_boxes, gt_boxes):
        tiled_pred_boxes = tf.expand_dims(pred_boxes, axis=1)
        tiled_pred_boxes = tf.tile(tiled_pred_boxes, (1, tf.shape(gt_boxes)[0], 1))
        ious = broadcast_iou(gt_boxes[..., :4], tiled_pred_boxes)
        best_bbox_idx = tf.argmax(ious, axis=-1)
        best_gt_boxes = tf.gather(gt_boxes, best_bbox_idx)
        is_overlap = tf.reduce_max(ious, axis=-1) > self.config['train']['roi_overlap_threshold']

        gt_boxes_y = (best_gt_boxes[..., 0] + best_gt_boxes[..., 2]) / 2
        gt_boxes_x = (best_gt_boxes[..., 1] + best_gt_boxes[..., 3]) / 2
        gt_boxes_h = best_gt_boxes[..., 2] - best_gt_boxes[..., 0]
        gt_boxes_w = best_gt_boxes[..., 3] - best_gt_boxes[..., 1]

        pred_boxes_y = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
        pred_boxes_x = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
        pred_boxes_h = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_boxes_w = pred_boxes[..., 3] - pred_boxes[..., 1]

        ty = (gt_boxes_y - pred_boxes_y) / pred_boxes_h
        tx = (gt_boxes_x - pred_boxes_x) / pred_boxes_w
        th = tf.math.log(gt_boxes_h / pred_boxes_h)
        tw = tf.math.log(gt_boxes_w / pred_boxes_w)

        target_cls = best_gt_boxes[..., 4]
        target_cls = tf.cast(target_cls, tf.int32)
        bg_indices = tf.where(tf.logical_not(is_overlap))
        target_cls = tf.tensor_scatter_nd_update(target_cls,
                                                 bg_indices,
                                                 tf.zeros((tf.shape(bg_indices)[0],),
                                                          dtype=tf.int32) + self.config['data']['num_classes'])  # BG
        one_hot_cls = tf.one_hot(target_cls, depth=self.config['data']['num_classes'] + 1)

        y = tf.concat([tf.expand_dims(ty, axis=-1),
                       tf.expand_dims(tx, axis=-1),
                       tf.expand_dims(th, axis=-1),
                       tf.expand_dims(tw, axis=-1),
                       tf.cast(one_hot_cls, tf.float32)], axis=-1)
        return y



    def _sample_rois(self, n_roi_xs, n_roi_ys):
        ret_xs = []
        ret_ys = []
        for roi_xs, roi_ys in zip(n_roi_xs, n_roi_ys):
            not_bg = tf.not_equal(roi_ys[..., -1], 1)
            is_pos = not_bg
            is_neg = tf.logical_not(is_pos)
            pos_xs = tf.gather_nd(roi_xs, tf.where(is_pos))
            pos_ys = tf.gather_nd(roi_ys, tf.where(is_pos))
            neg_xs = tf.gather_nd(roi_xs, tf.where(is_neg))
            neg_ys = tf.gather_nd(roi_ys, tf.where(is_neg))
            pos_cnt = tf.shape(pos_xs)[0]
            neg_cnt = tf.shape(neg_xs)[0]
            # https://github.com/tensorflow/tensorflow/issues/26608
            # random_idx = tf.random.shuffle(tf.expand_dims(tf.range(neg_cnt), axis=-1))
            # random_idx = random_idx[:pos_cnt]
            # neg_xs = tf.gather_nd(neg_xs, random_idx)
            # neg_ys = tf.gather_nd(neg_ys, random_idx)
            neg_xs = neg_xs[:pos_cnt]
            neg_ys = neg_ys[:pos_cnt]
            ret_roi_xs = tf.concat([pos_xs, neg_xs], axis=0)
            ret_roi_ys = tf.concat([pos_ys, neg_ys], axis=0)
            ret_xs.append(ret_roi_xs)
            ret_ys.append(ret_roi_ys)
        return ret_xs, ret_ys

    def _rpn_loss(self, rpn_ys, rpn_pred_objs, rpn_pred_regrs):
        ret = 0
        for rpn_y, rpn_pred_obj, rpn_pred_regr in zip(rpn_ys, rpn_pred_objs, rpn_pred_regrs):
            loss = rpn_loss(rpn_y, rpn_pred_obj, rpn_pred_regr)
            ret += loss
        return ret


    def _anchor_boxes_like(self, target_maps):
        ret = []
        for target_map in target_maps:
            target_shape = tf.cast(tf.shape(target_map), tf.float32)
            N = target_shape[0]
            H = target_shape[1]
            W = target_shape[2]
            anchors_coord = make_anchor_coords(H, W, self.anchors)
            anchors_coord = tf.expand_dims(anchors_coord, axis=0)
            anchors_coord = tf.tile(anchors_coord, (N, 1, 1, 1, 1))
            ret.append(anchors_coord)
        return ret

    def _apply_regr(self, boxes_layers, regr_layers):
        ret = []
        for boxes, regr in zip(boxes_layers, regr_layers):
            box_center_y = (boxes[..., 0] + boxes[..., 2]) / 2
            box_center_x = (boxes[..., 1] + boxes[..., 3]) / 2
            box_h = boxes[..., 2] - boxes[..., 0]
            box_w = boxes[..., 3] - boxes[..., 1]

            regr = tf.reshape(regr, tf.shape(boxes))

            new_y = regr[..., 0] * box_h + box_center_y
            new_x = regr[..., 1] * box_w + box_center_x
            new_h = tf.math.exp(regr[..., 2]) * box_h
            new_w = tf.math.exp(regr[..., 3]) * box_w
            coords = tf.stack([new_y - new_h / 2,
                               new_x - new_w / 2,
                               new_y + new_h / 2,
                               new_x + new_w / 2], axis=-1)
            ret.append(coords)
        return ret

    def _roi_align(self, rpn_features_layers, rpn_boxes_layers):
        rois = []
        for rpn_features, rpn_boxes in zip(rpn_features_layers, rpn_boxes_layers):
            for n in range(self.config['train']['rpn_batch_size']):
                nth_boxes = rpn_boxes[n]
                num_boxes = tf.shape(nth_boxes)[0]
                box_indices = tf.constant(n, shape=(1,))
                box_indices = tf.tile(box_indices, tf.expand_dims(num_boxes, axis=0))
                roi = tf.image.crop_and_resize(rpn_features,
                                               nth_boxes,
                                               box_indices,
                                               (self._roi_size, self._roi_size))
                rois.append(roi)
        return rois

    def _nms(self, rpn_boxes_layers, rpn_objs_layers, iou_threshold=0.5):
        ret = []
        for rpn_boxes, rpn_objs in zip(rpn_boxes_layers, rpn_objs_layers):
            boxes_per_batch = []
            for n in range(self.config['train']['rpn_batch_size']):
                nth_boxes = rpn_boxes[n]
                nth_objs = rpn_objs[n]
                nth_boxes = tf.reshape(nth_boxes, (-1, 4))
                nth_objs = tf.reshape(nth_objs, (-1, ))
                selected_indices = tf.image.non_max_suppression(nth_boxes, nth_objs, self.config['train']['max_num_rois'], iou_threshold=iou_threshold)
                selected_boxes = tf.gather(nth_boxes, selected_indices)
                boxes_per_batch.append(selected_boxes)
            ret.append(boxes_per_batch)
        return ret

    def _roi_target(self, rpn_boxes_layers, gt_boxes):
        ret = []
        for rpn_boxes in rpn_boxes_layers:
            for n in range(self.config['train']['rpn_batch_size']):
                roi_regr = self._calc_roi_target(rpn_boxes[n], gt_boxes[n])
                ret.append(roi_regr)
        return ret


    def _classify(self, roi_xs):
        roi_clses = []
        roi_regrs = []
        for roi_x in roi_xs:
            roi_cls, roi_regr = self.classifier(roi_x)
            roi_clses.append(roi_cls)
            roi_regrs.append(roi_regr)
        return roi_clses, roi_regrs

    def _roi_loss(self,  roi_ys, roi_clses, roi_regrs):
        losses = 0
        for roi_y, roi_cls, roi_regr in zip(roi_ys, roi_clses, roi_regrs):
            loss = roi_loss(roi_y, roi_cls, roi_regr, self.config['data']['num_classes'])
            losses += loss
        return losses

    @tf.function
    def _train_step(self, x, rpn_ys, gt_boxes):
        with tf.GradientTape() as tape:
            rpn_objs, rpn_regrs, rpn_features = self.rpn(x)
            rpn_loss = self._rpn_loss(rpn_ys, rpn_objs, rpn_regrs)
            anchor_boxes = self._anchor_boxes_like(rpn_regrs)
            rpn_boxes = self._apply_regr(anchor_boxes, rpn_regrs)
            suppressed_boxes = self._nms(rpn_boxes, rpn_objs)
            roi_xs = self._roi_align(rpn_features, suppressed_boxes)
            roi_ys = self._roi_target(suppressed_boxes, gt_boxes)
            roi_xs, roi_ys = self._sample_rois(roi_xs, roi_ys)
            roi_clses, roi_regrs = self._classify(roi_xs)
            roi_loss = self._roi_loss(roi_ys, roi_clses, roi_regrs)
            loss = rpn_loss+roi_loss
        grad = tape.gradient(loss, self.rpn.trainable_variables + self.classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.rpn.trainable_variables + self.classifier.trainable_variables))
        return rpn_loss+roi_loss

    def train(self):
        self._init_train_context()
        best_loss = np.inf
        for epoch in range(self.config['train']['epoch']):
            for step, (x, (rpn_y, gt_boxes)) in enumerate(self.dataset):
                loss = self._train_step(x, rpn_y, gt_boxes)
                self.train_loss(loss)
            print('Epoch {}: Loss: {}'.format(epoch, self.train_loss.result()))

            self.ckpt_manager.save()
            if best_loss > self.train_loss.result().numpy():
                print('Saved model')
                self.saver.save_weights(self.config['train']['model_file'])
                best_loss = self.train_loss.result().numpy()
            self.train_loss.reset_states()

    def _select_objs(self, boxes, regrs, clses):
        obj_clses = []
        obj_regrs = []
        obj_boxes = []

        for box, regr, cls in zip(boxes, regrs, clses):
            pred_cls = tf.argmax(cls, axis=-1)
            obj_idx = tf.where(tf.math.not_equal(pred_cls, self.config['data']['num_classes']))

            regr = tf.reshape(regr, (-1, self.config['data']['num_classes'], 4))
            regr_idx = tf.concat([obj_idx, tf.gather(pred_cls, obj_idx)], axis=-1)

            obj_regr = tf.gather_nd(regr, regr_idx)
            obj_cls = tf.gather(pred_cls, obj_idx)
            obj_box = tf.gather_nd(box, obj_idx)

            obj_regrs.append(obj_regr)
            obj_clses.append(obj_cls)
            obj_boxes.append(obj_box)
        return obj_boxes, obj_regrs, obj_clses

    def test(self, path):
        self.rpn.build(input_shape=(None, 416, 416, 3))
        self.classifier.build(input_shape=(None, self._roi_size, self._roi_size, 512))
        self.saver.load_weights(self.config['train']['model_file'])

        img = tf.image.decode_png(open(path, 'rb').read(), channels=3)
        img = tf.image.resize(img, (416, 416))
        img = tf.expand_dims(img, axis=0)
        rpn_objs, rpn_regrs, rpn_features = self.rpn(img)
        anchor_boxes = self._anchor_boxes_like(rpn_regrs)
        rpn_boxes = self._apply_regr(anchor_boxes, rpn_regrs)
        suppressed_boxes = self._nms(rpn_boxes, rpn_objs, iou_threshold=0.7)
        roi_xs = self._roi_align(rpn_features, suppressed_boxes)
        roi_clses, roi_regrs = self._classify(roi_xs)
        obj_boxes, obj_regrs, obj_clses = self._select_objs(list(itertools.chain.from_iterable(suppressed_boxes)), roi_regrs, roi_clses)
        obj_boxes = self._apply_regr(obj_boxes, obj_regrs)
        box_for_drawing = tf.expand_dims(tf.concat(obj_boxes, axis=0), axis=0)
        img_with_boxes = tf.image.draw_bounding_boxes(img, box_for_drawing, colors=np.array([[1., 1., 1., 1.]]))
        tf.io.write_file('test.jpg', tf.image.encode_jpeg(tf.squeeze(tf.cast(img_with_boxes, tf.uint8))))
