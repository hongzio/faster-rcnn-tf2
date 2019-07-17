import tensorflow as tf

class VGG(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        self.block1_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        self.block2_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        self.block3_conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        self.block3_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        self.block4_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        self.block4_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        self.block5_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        self.block5_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        self.block5_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.reduce5 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='reduce5')
        self.up5 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.reduce4 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='reduce4')
        self.up4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.add4 = tf.keras.layers.Add()

        self.reduce3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='reduce3')
        self.up3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.add3 = tf.keras.layers.Add()

        self.reduce2 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same', name='reduce2')
        self.add2 = tf.keras.layers.Add()


    def call(self, x, **kwargs):
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = C2 = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = C3 = self.block3_pool(x)

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = C4 = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        C5 = self.block5_pool(x)

        P5 = self.reduce5(C5)
        P4 = self.add4([self.up5(P5), self.reduce4(C4)])
        P3 = self.add3([self.up4(P4), self.reduce3(C3)])
        P2 = self.add2([self.up3(P3), self.reduce2(C2)])
        return [P2, P3, P4, P5]

    def calc_output_size(self, input_shape):
        return [input_shape / 4, input_shape / 8, input_shape / 16, input_shape / 32]

    def pyramid_depth(self):
        return 4