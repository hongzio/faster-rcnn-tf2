import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError


def _hard_negative_sampling(losses, valid_mask, overlap_mask):
    neg_mask = tf.math.logical_and(tf.equal(valid_mask, 1.0), tf.not_equal(overlap_mask, 1.0))
    pos_idx = tf.where(tf.math.logical_not(neg_mask))
    neg_losses = tf.tensor_scatter_nd_update(losses, pos_idx, tf.zeros((tf.shape(pos_idx)[0], )))
    flat_neg_losses = tf.reshape(neg_losses, (-1, ))
    sorted_losses = tf.sort(flat_neg_losses, direction='DESCENDING')
    pos_cnt = tf.math.count_nonzero(overlap_mask)
    neg_cnt = tf.math.maximum(pos_cnt, 8)
    threshold = sorted_losses[neg_cnt]
    neg_samples = tf.math.greater_equal(losses, threshold)
    neg_sample_idx = tf.where(neg_samples)
    neg_sample_idx = tf.random.shuffle(neg_sample_idx)
    ret = tf.zeros(tf.shape(valid_mask))
    try:
        ret = tf.tensor_scatter_nd_update(ret, neg_sample_idx[:neg_cnt], tf.ones((neg_cnt, )))
    except InvalidArgumentError as e:
        print(e)
    return ret

eps = 1e-6
def rpn_loss(rpn_y, rpn_pred_obj, rpn_pred_regr):
    valid_mask = rpn_y[..., 0]
    overlap_mask = rpn_y[..., 1]

    rpn_pred_obj = tf.expand_dims(rpn_pred_obj, axis=-1)
    overlap_mask_reshaped = tf.reshape(overlap_mask, tf.shape(rpn_pred_obj))
    rpn_obj_loss = valid_mask * tf.keras.losses.binary_crossentropy(overlap_mask_reshaped, rpn_pred_obj)

    negative_mask = _hard_negative_sampling(rpn_obj_loss, valid_mask, overlap_mask)
    train_mask = tf.cast(tf.logical_or(tf.cast(overlap_mask, tf.bool), tf.cast(negative_mask, tf.bool)), tf.float32)
    rpn_obj_loss = train_mask * rpn_obj_loss

    rpn_obj_loss = tf.math.reduce_sum(rpn_obj_loss) / (tf.math.reduce_sum(train_mask)+eps)


    rpn_regress_pred = tf.reshape(rpn_pred_regr, rpn_y[..., 2:].shape)
    rpn_regress_loss = tf.math.square(rpn_y[..., 2:] - rpn_regress_pred)

    tiled_train_mask = tf.tile(tf.expand_dims(train_mask, axis=-1), (1, 1, 1, 1, 4))
    rpn_regress_loss = tf.math.reduce_sum(tiled_train_mask * rpn_regress_loss) / (tf.math.reduce_sum(tiled_train_mask) + eps)
    return rpn_obj_loss + rpn_regress_loss


def roi_loss(y_true, pred_cls, pred_regr, num_classes):
    true_regr = y_true[..., :4]
    true_cls = y_true[..., 4:]

    pred_regr = tf.reshape(pred_regr, (tf.shape(pred_regr)[0], num_classes, 4))

    mask = true_cls[..., :-1]
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, (1, 1, 4))

    true_regr = tf.expand_dims(true_regr, axis=1)
    true_regr = tf.tile(true_regr, (1, num_classes, 1))

    cls_loss = tf.losses.categorical_crossentropy(true_cls, pred_cls)
    cls_loss = tf.reduce_sum(cls_loss)

    regr_loss = tf.reduce_sum(tf.math.square(true_regr - pred_regr) * mask) / (tf.reduce_sum(mask) + eps)

    return cls_loss + regr_loss