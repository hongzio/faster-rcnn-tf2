import tensorflow as tf

class RPN(tf.keras.models.Model):
    def __init__(self, num_anchors, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone(name='backbone')
        self.rpn_outs = []
        for i in range(self.backbone.pyramid_depth()):
            rpn_conv = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                                               kernel_initializer='normal', name='rpn_conv1_' + str(i))
            rpn_out_obj = tf.keras.layers.Conv2D(num_anchors, (3, 3), padding='same', activation='sigmoid',
                                                 kernel_initializer='uniform', name='rpn_out_obj_' + str(i))
            rpn_out_regress = tf.keras.layers.Conv2D(num_anchors * 4, (3, 3), padding='same',
                                                     activation='linear', kernel_initializer='zero',
                                                     name='rpn_out_regress_' + str(i))
            self.rpn_outs.append([rpn_conv, rpn_out_obj, rpn_out_regress])

    def call(self, x, **kwargs):
        xs = self.backbone(x)
        x_features = []
        x_objs = []
        x_regressions = []
        for x, rpn_out in zip(xs, self.rpn_outs):
            x_feature = rpn_out[0](x)
            x_obj = rpn_out[1](x_feature)
            x_regress = rpn_out[2](x_feature)
            x_features.append(x_feature)
            x_objs.append(x_obj)
            x_regressions.append(x_regress)
        return x_objs, x_regressions, x_features
