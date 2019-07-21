import tensorflow as tf
def _hard_negative_sampling(losses, valid_mask, overlap_mask):
    neg_mask = tf.math.logical_and(tf.equal(valid_mask, 1.0), tf.not_equal(overlap_mask, 1.0))
    pos_idx = tf.where(tf.math.logical_not(neg_mask))
    neg_losses = tf.tensor_scatter_nd_update(losses, pos_idx, tf.zeros((tf.shape(pos_idx)[0], )))
    flat_neg_losses = tf.reshape(neg_losses, (-1, ))
    sorted_losses = tf.sort(flat_neg_losses, direction='DESCENDING')
    pos_cnt = tf.math.count_nonzero(overlap_mask)
    neg_cnt = pos_cnt
    threshold = sorted_losses[neg_cnt]
    neg_samples = tf.math.greater_equal(losses, threshold)
    neg_sample_idx = tf.where(neg_samples)
    neg_sample_idx = tf.random.shuffle(neg_sample_idx)
    ret = tf.zeros(tf.shape(valid_mask))
    ret = tf.tensor_scatter_nd_update(ret, neg_sample_idx[:neg_cnt], tf.ones((neg_cnt, )))
    return ret

eps = 1e-6
def rpn_loss(y_true, rpn_out):
    valid_mask = y_true[..., 0]
    overlap_mask = y_true[..., 1]

    rpn_obj_pred = rpn_out[0]
    rpn_obj_pred = tf.expand_dims(rpn_obj_pred, axis=-1)
    overlap_mask_reshaped = tf.reshape(overlap_mask, tf.shape(rpn_obj_pred))
    rpn_obj_loss = valid_mask * tf.keras.losses.binary_crossentropy(overlap_mask_reshaped, rpn_obj_pred)

    negative_mask = _hard_negative_sampling(rpn_obj_loss, valid_mask, overlap_mask)
    train_mask = tf.cast(tf.logical_or(tf.cast(overlap_mask, tf.bool), tf.cast(negative_mask, tf.bool)), tf.float32)
    rpn_obj_loss = train_mask * rpn_obj_loss

    rpn_obj_loss = tf.math.reduce_sum(rpn_obj_loss) / (tf.math.reduce_sum(train_mask)+eps)


    rpn_regress_pred = tf.reshape(rpn_out[1], y_true[..., 2:].shape)
    rpn_regress_loss = tf.math.square(y_true[..., 2:] - rpn_regress_pred)

    tiled_train_mask = tf.tile(tf.expand_dims(train_mask, axis=-1), (1, 1, 1, 1, 4))
    rpn_regress_loss = tf.math.reduce_sum(tiled_train_mask * rpn_regress_loss) / tf.math.reduce_sum(tiled_train_mask + eps)
    return rpn_obj_loss + rpn_regress_loss


def roi_loss(y_true, y_pred, num_classes):
    is_valid = y_true[..., 0]
    true_regr = y_true[..., 1:5]
    true_cls = y_true[..., 5:]

    pred_cls = y_pred[0] # (None, num_classes+1)
    pred_regr = y_pred[1] # (None, num_classes)
    pred_regr = tf.reshape(pred_regr, (tf.shape(pred_regr)[0], num_classes, 4))

    mask = true_cls[..., :-1]
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.tile(mask, (1, 1, 4))

    true_regr = tf.expand_dims(true_regr, axis=1)
    true_regr = tf.tile(true_regr, (1, num_classes, 1))

    cls_loss = tf.losses.categorical_crossentropy(true_cls, pred_cls)
    cls_loss = tf.reduce_sum(cls_loss) / tf.reduce_sum(mask + eps)

    regr_loss = tf.reduce_sum(tf.math.square(true_regr - pred_regr) * mask) / tf.reduce_sum(mask + eps)

    return tf.reduce_sum(cls_loss) + tf.reduce_sum(regr_loss)