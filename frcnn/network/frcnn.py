from copy import deepcopy

import tensorflow as tf

from frcnn.loss import rpn_loss, roi_loss
from frcnn.network.classifier import RoiClassifier
from frcnn.network.rpn import RPN
from frcnn.util.anchor import regress_to_coord, broadcast_iou


class FasterRCNNModel(tf.keras.models.Model):

    def __init__(self, anchors, backbone, rpn_batch_size, roi_batch_size, num_classes, max_num_rois=300, roi_size=7,
                 roi_overlap_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchors
        self.max_num_rois = tf.constant(max_num_rois)
        self.rpn_batch_size = rpn_batch_size
        self.roi_batch_size = roi_batch_size
        self.roi_size = roi_size
        self.roi_overlap_threshold = roi_overlap_threshold
        self.num_classes = num_classes
        self.rpn = RPN(len(anchors), backbone)
        self.classifier = RoiClassifier(self.num_classes)

    def _calc_roi_target(self, pred_boxes, gt_boxes):
        tiled_pred_boxes = tf.expand_dims(pred_boxes, axis=1)
        tiled_pred_boxes = tf.tile(tiled_pred_boxes, (1, tf.shape(gt_boxes)[0], 1))
        ious = broadcast_iou(gt_boxes[..., :4], tiled_pred_boxes)
        best_bbox_idx = tf.argmax(ious, axis=-1)
        best_gt_boxes = tf.gather(gt_boxes, best_bbox_idx)
        is_overlap = tf.reduce_max(ious, axis=-1) > self.roi_overlap_threshold

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
                                                          dtype=tf.int32) + self.num_classes)  # BG
        one_hot_cls = tf.one_hot(target_cls, depth=self.num_classes + 1)

        y = tf.concat([tf.expand_dims(ty, axis=-1),
                       tf.expand_dims(tx, axis=-1),
                       tf.expand_dims(th, axis=-1),
                       tf.expand_dims(tw, axis=-1),
                       tf.cast(one_hot_cls, tf.float32)], axis=-1)
        return y

    def _roi_align(self, rpn_outs, gt_boxes):
        rois = []
        ys = []
        for rpn_out in rpn_outs:
            rpn_score = rpn_out[0]
            rpn_regress = rpn_out[1]
            rpn_feature_map = rpn_out[2]
            boxes = regress_to_coord(rpn_regress, self.anchors)
            for n in range(self.rpn_batch_size):
                nth_boxes = tf.reshape(boxes[n], (-1, 4))
                nth_scores = tf.reshape(rpn_score[n], (-1,))
                selected_nth_indices = tf.image.non_max_suppression(nth_boxes, nth_scores, self.max_num_rois)
                selected_nth_boxes = tf.gather(nth_boxes, selected_nth_indices)
                is_valid = tf.ones((tf.shape(selected_nth_boxes)[0],))
                num_boxes = tf.shape(selected_nth_boxes)[0]
                box_indices = tf.constant(n, shape=(1, ))
                box_indices = tf.tile(box_indices, tf.expand_dims(num_boxes, axis=0))
                roi = tf.image.crop_and_resize(rpn_feature_map,
                                               selected_nth_boxes,
                                               box_indices,
                                               (self.roi_size, self.roi_size))
                y = self._calc_roi_target(selected_nth_boxes, gt_boxes[n])
                y = tf.concat([tf.expand_dims(is_valid, axis=-1), y], axis=1)
                paddings = tf.zeros((4, 2), dtype=tf.int32)
                paddings = tf.tensor_scatter_nd_update(paddings,
                                                       tf.constant([[0, 1]]),
                                                       tf.expand_dims(self.max_num_rois - num_boxes, axis=0))
                roi = tf.pad(roi, paddings)
                y = tf.pad(y, paddings[0:2])
                rois.append(roi)
                ys.append(y)
        return tf.convert_to_tensor(rois), tf.convert_to_tensor(ys)

    def sample_rois(self, roi_xs, roi_ys):
        not_bg = tf.not_equal(roi_ys[..., -1], 1)
        is_valid = tf.equal(roi_ys[..., 0], 1)
        is_pos = tf.logical_and(not_bg, is_valid)
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
        return ret_roi_xs, ret_roi_ys

    def call(self, x, rpn_y, gt_boxes, **kwargs):
        losses = 0
        rpn_outs = self.rpn(x)
        for y, rpn_out in zip(rpn_y, rpn_outs):
            loss = rpn_loss(y, rpn_out)
            losses += loss
        roi_xs, roi_ys = self._roi_align(rpn_outs, gt_boxes)
        roi_xs = tf.reshape(roi_xs, (-1, 7, 7, 512))
        roi_ys = tf.reshape(roi_ys, (-1, 106))
        roi_xs, roi_ys = self.sample_rois(roi_xs, roi_ys)
        for i in range(0, tf.shape(roi_xs)[0], self.roi_batch_size):
            roi_x = roi_xs[i:i+self.roi_batch_size]
            roi_y = roi_ys[i:i+self.roi_batch_size]
            roi_pred = self.classifier(roi_x)
            loss = roi_loss(roi_y, roi_pred, self.num_classes)
            losses += loss
        return losses
