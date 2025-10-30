import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19

IMG_SIZE = 256

# VGG model for perceptual loss
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg.trainable = False
vgg_model = models.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

def hybrid_loss(y_true, y_pred):
    mse = 0.6 * tf.reduce_mean(tf.square(y_true - y_pred))
    y_true_scaled = y_true * 255.0
    y_pred_scaled = y_pred * 255.0
    vgg_loss = 0.4 * tf.reduce_mean(tf.square(vgg_model(y_true_scaled) - vgg_model(y_pred_scaled)))
    return mse + vgg_loss

def build_unet_denoiser():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    e1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(e2)

    # Bottleneck
    b = layers.Conv2D(256, (3,3), dilation_rate=2, padding='same')(p2)
    b = layers.LeakyReLU(alpha=0.1)(b)
    b = layers.Conv2D(256, (3,3), dilation_rate=2, padding='same')(b)
    b = layers.LeakyReLU(alpha=0.1)(b)

    # Decoder
    d2 = up_block(b, e2, 128)
    d1 = up_block(d2, e1, 64)

    # Output
    output = layers.Conv2D(3, (3,3), padding='same')(d1)
    output = layers.Add()([inputs, output * 0.7])
    output = layers.Activation('sigmoid')(output)

    return models.Model(inputs, output)

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def up_block(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.concatenate([x, skip])
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x
