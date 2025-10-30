import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from model import build_unet_denoiser, hybrid_loss
from utils import load_images_from_folder, add_mixed_noise, evaluate_denoiser

BATCH_SIZE = 16
EPOCHS = 100
INIT_LR = 3e-4

# Load data
x_train = load_images_from_folder("data/train")
x_test = load_images_from_folder("data/test")

x_train_noisy = add_mixed_noise(x_train)
x_test_noisy = add_mixed_noise(x_test)

# Model setup
autoencoder = build_unet_denoiser()
autoencoder.compile(
    optimizer=optimizers.Adam(learning_rate=INIT_LR),
    loss=hybrid_loss,
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
)

# Callbacks
cb = [
    callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    callbacks.ModelCheckpoint('results/best_denoiser.h5', save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
]

# Train
history = autoencoder.fit(
    x_train_noisy, x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_noisy, x_test),
    callbacks=cb
)

# Evaluate
evaluate_denoiser(autoencoder, x_test_noisy, x_test)
