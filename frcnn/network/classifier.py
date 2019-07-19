import tensorflow as tf
class RoiClassifier(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1')
        self.do1 = tf.keras.layers.Dropout(0.1, name='do1')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', name='fc2')
        self.do2 = tf.keras.layers.Dropout(0.1, name='do2')

        self.out_cls = tf.keras.layers.Dense(num_classes+1, activation='softmax', kernel_initializer='zero',
                                             name='out_cls')
        self.out_regr = tf.keras.layers.Dense(4 * num_classes, activation='linear', kernel_initializer='zero',
                                              name='out_regr')

    def call(self, x, **kwargs):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.do2(x)

        o_cls = self.out_cls(x)
        o_regr = self.out_regr(x)

        return (o_cls, o_regr)