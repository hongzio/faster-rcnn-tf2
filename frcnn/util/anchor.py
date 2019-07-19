import tensorflow as tf
import numpy as np
Y = 0
X = 1

def make_anchors(config):
    a_sizes = np.array(config['anchor']['sizes'])
    a_ratios = np.array(config['anchor']['ratios'])
    return tf.convert_to_tensor([a_size*a_ratio for a_size in a_sizes for a_ratio in a_ratios], dtype=tf.float32)

def anchor_coords_along(way, centers, anchors):
    """

    :param way: 0=axis y, 1=axis x
    :param centers: center coordinates of each grid
    :param anchors: [[h1, w1], [h2, w2], ...] anchor list
    :return:
    """
    num_anchors = tf.shape(anchors)[0]

    centers = tf.cast(centers, tf.float32)
    centers = tf.expand_dims(centers, axis=-1)  # anchors
    centers = tf.expand_dims(centers, axis=-1)  # coords
    centers = tf.tile(centers, [1, 1, num_anchors, 2])
    anchors_delta = tf.stack([(-anchors / 2)[..., way], (anchors / 2)[..., way]], axis=-1)
    return centers + anchors_delta + 0.5


def broadcast_iou(ref_box, anchor_box):
    """

    :param ref_box: (..., 4(y1, x1, y2, x2))
    :param anchor_box: (..., N, 4(y1, x1, y2, x2))
    :return: (..., N)
    """
    new_shape = tf.broadcast_dynamic_shape(tf.shape(ref_box), tf.shape(anchor_box))
    ref_box = tf.broadcast_to(ref_box, new_shape)
    anchor_box = tf.broadcast_to(anchor_box, new_shape)

    int_w = tf.minimum(ref_box[..., 3], anchor_box[..., 3]) - tf.maximum(ref_box[..., 1], anchor_box[..., 1])
    int_w = tf.maximum(int_w, 0)
    int_h = tf.minimum(ref_box[..., 2], anchor_box[..., 2]) - tf.maximum(ref_box[..., 0], anchor_box[..., 0])
    int_h = tf.maximum(int_h, 0)
    int_area = int_w * int_h
    ref_box_area = (ref_box[..., 2] - ref_box[..., 0]) * (ref_box[..., 3] - ref_box[..., 1])
    anchor_box_area = (anchor_box[..., 2] - anchor_box[..., 0]) * (anchor_box[..., 3] - anchor_box[..., 1])
    return int_area / (ref_box_area + anchor_box_area - int_area + 1e-6)



def regress_to_coord(regress, anchors):
    N, H, W, A = tf.cast(tf.shape(regress), tf.float32)
    anchors_coord = make_anchor_coords(H, W, anchors)
    anchors_coord = tf.expand_dims(anchors_coord, axis=0)
    anchors_coord = tf.tile(anchors_coord, (N, 1, 1, 1, 1))
    anchor_center_y = (anchors_coord[..., 0] + anchors_coord[..., 2]) / 2
    anchor_center_x = (anchors_coord[..., 1] + anchors_coord[..., 3]) / 2
    anchor_h = anchors_coord[..., 2] - anchors_coord[..., 0]
    anchor_w = anchors_coord[..., 3] - anchors_coord[..., 1]

    regress = tf.reshape(regress, tf.shape(anchors_coord))

    anchor_new_y = regress[..., 0] * anchor_h + anchor_center_y
    anchor_new_x = regress[..., 1] * anchor_w + anchor_center_x
    anchor_new_h = tf.math.exp(regress[..., 2]) * anchor_h
    anchor_new_w = tf.math.exp(regress[..., 3]) * anchor_w
    coords = tf.stack([anchor_new_y - anchor_new_h / 2,
                       anchor_new_x - anchor_new_w / 2,
                       anchor_new_y + anchor_new_h / 2,
                       anchor_new_x + anchor_new_w / 2], axis=-1)
    return coords

def transform(bboxes, anchors, output_sizes, min_iou, max_iou, max_num_boxes=100):
    # Padding bboxes
    paddings = [[0, max_num_boxes - tf.shape(bboxes)[0]],
                [0, 0]]  # up, *down*, left, right. 100 - The maximum number of objects
    bboxes = tf.pad(bboxes, paddings)
    ys = []
    for output_size in output_sizes:
        # anchors' coordinate
        H = tf.cast(output_size[0], tf.float32)
        W = tf.cast(output_size[1], tf.float32)
        anchors_coord = make_anchor_coords(H, W, anchors)
        anchors_coord_per_bbox = tf.expand_dims(anchors_coord, axis=-2)  # bboxes
        anchors_coord_per_bbox = tf.tile(anchors_coord_per_bbox, [1, 1, 1, max_num_boxes, 1])

        ious = broadcast_iou(bboxes[..., :4], anchors_coord_per_bbox)

        out_of_bound_y = tf.logical_or(anchors_coord_per_bbox[..., 0] < 0, anchors_coord_per_bbox[..., 2] > 1)
        out_of_bound_x = tf.logical_or(anchors_coord_per_bbox[..., 1] < 0, anchors_coord_per_bbox[..., 3] > 1)
        out_of_bound = tf.logical_or(out_of_bound_y, out_of_bound_x)
        index = tf.where(out_of_bound)
        ious = tf.tensor_scatter_nd_update(ious, index, tf.zeros(tf.shape(index)[0]))

        best_bbox_idx = tf.cast(tf.math.argmax(ious, axis=-1), tf.int32)
        flat_best_bbox_rel_idx = tf.reshape(best_bbox_idx, [-1])
        flat_base_idx = tf.range(tf.cast(H * W * len(anchors), tf.int32), dtype=tf.int32)
        flat_best_bbox_abs_idx = tf.stack([flat_base_idx, flat_best_bbox_rel_idx], axis=1)

        bboxes_per_grid = tf.broadcast_to(bboxes, (H, W, len(anchors), max_num_boxes, 5))

        flat_bboxes = tf.reshape(bboxes_per_grid, (-1, max_num_boxes, 5))
        flat_best_bboxes = tf.gather_nd(flat_bboxes, flat_best_bbox_abs_idx)
        best_bboxes = tf.reshape(flat_best_bboxes, (H, W, len(anchors), 5))

        bbox_center_y = (best_bboxes[..., 0] + best_bboxes[..., 2]) / 2
        bbox_center_x = (best_bboxes[..., 1] + best_bboxes[..., 3]) / 2
        anchor_center_y = (anchors_coord[..., 0] + anchors_coord[..., 2]) / 2
        anchor_center_x = (anchors_coord[..., 1] + anchors_coord[..., 3]) / 2
        bbox_h = best_bboxes[..., 2] - best_bboxes[..., 0]
        bbox_w = best_bboxes[..., 3] - best_bboxes[..., 1]
        anchor_h = anchors_coord[..., 2] - anchors_coord[..., 0]
        anchor_w = anchors_coord[..., 3] - anchors_coord[..., 1]
        tx = (bbox_center_x - anchor_center_x) / anchor_w
        ty = (bbox_center_y - anchor_center_y) / anchor_h
        tw = tf.math.log(bbox_w / anchor_w)
        th = tf.math.log(bbox_h / anchor_h)
        regress = tf.stack([tx, ty, tw, th], axis=-1)

        valid = tf.math.logical_or(ious < min_iou, max_iou < ious)
        valid = tf.math.reduce_any(valid, axis=-1)
        valid = tf.cast(valid, tf.float32)
        valid = tf.expand_dims(valid, axis=-1)

        overlap = max_iou < ious
        overlap = tf.math.reduce_any(overlap, axis=-1)
        overlap = tf.cast(overlap, tf.float32)
        overlap = tf.expand_dims(overlap, axis=-1)

        y = tf.concat([valid, overlap, regress], axis=-1)
        ys.append(y)
    return tuple(ys), bboxes


def make_anchor_coords(H, W, anchors):
    y_center, x_center = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
    anchors_y = anchor_coords_along(Y, y_center, anchors)
    anchors_y_ratio = anchors_y / H
    anchors_x = anchor_coords_along(X, x_center, anchors)
    anchors_x_ratio = anchors_x / W
    anchors_coord = tf.concat([anchors_y_ratio[..., 0:1],
                               anchors_x_ratio[..., 0:1],
                               anchors_y_ratio[..., 1:2],
                               anchors_x_ratio[..., 1:2]], axis=-1)
    return anchors_coord
