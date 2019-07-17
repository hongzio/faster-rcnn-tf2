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

def rpn_loss(y_true, rpn_out):
    eps = 1e-6
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

    tiled_overlap_mask = tf.reshape(tf.tile(tf.reshape(overlap_mask, (-1, 1)), (1, tf.shape(rpn_regress_loss)[-1])),
                                    tf.shape(rpn_regress_loss))
    rpn_regress_loss = tf.math.reduce_sum(tiled_overlap_mask * rpn_regress_loss) / tf.math.reduce_sum(tiled_overlap_mask + eps)
    return rpn_obj_loss + rpn_regress_loss
