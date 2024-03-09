import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
from sklearn.model_selection import train_test_split
import argparse
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from keras.datasets import mnist


def load_input_data():
    # Load input data using mnist dataset for illustration
    (train_X, _), (test_X, _) = mnist.load_data()

    # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)

    # Normalize the pixel values
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    return train_X, test_X



# Load input data using the function load_input_data
input_images, input_labels = load_input_data()  # Call the correct function

# Split into train and validation sets
train_images, val_images = train_test_split(input_images, test_size=0.1, random_state=42)

# Autoencoder architecture - hyperparameters
parameter_latent_dim = 15
parameter_batch_size = 64
parameter_epochs = 3
parameter_convolution_layers = 2
parameter_filter_size = (3, 3)
parameter_pool_size = (2, 2)
parameter_drop_out_percent = 0.25

def build_autoencoder(input_shape, latent_dim, parameter_convolution_layers, filter_size, pool_size, parameter_drop_out_percent):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)

    neurons = 32

    #
    # Input layer of encoder (and autoencoder)
    #
    x = layers.Conv2D(neurons, filter_size, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(pool_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #
    # Inner layers - Encoder
    #
    for i in range(parameter_convolution_layers):
        layer_neurons = 2**(i+1)*neurons
        print("Adding a convolution layer in encoder with neurons: ", layer_neurons)
        x = layers.Conv2D(2**(i+1)*neurons, filter_size, activation='relu', padding='same')(x)
        x = layers.Dropout(parameter_drop_out_percent)(x)
        if i % 2 == 0:
            x = layers.MaxPooling2D(pool_size, padding='same')(x)
        x = layers.BatchNormalization()(x)

    # keep last layer dimensions ---> the decoder starts from those!!
    last_convolution_layer_shape_list = x.shape.as_list()
    last_convolution_layer_total_neurons = last_convolution_layer_shape_list[1] * last_convolution_layer_shape_list[2] * last_convolution_layer_shape_list[3]
    last_convolution_layer_shape = (
    last_convolution_layer_shape_list[1], last_convolution_layer_shape_list[2], last_convolution_layer_shape_list[3])

    #
    # Inner layers - latent vector
    #
    x = layers.Flatten()(x)
    latent_vector = x = layers.Dense(latent_dim, activation='relu')(x)

    #
    # Inner layers - Decoder - input layer
    #
    x = layers.Dense(last_convolution_layer_total_neurons, activation='relu')(x)
    x = layers.Reshape(last_convolution_layer_shape)(x) 
    #
    # Inner layers - Decoder
    #
    neurons = 2**(parameter_convolution_layers - 2)*neurons

    print(neurons)

    for i in range(parameter_convolution_layers):
        print("Adding a convolution layer in decoder with neurons: ", layer_neurons)
        x = layers.Conv2D(layer_neurons, filter_size, activation='relu', padding='same')(x)
        x = layers.Dropout(parameter_drop_out_percent)(x)
        if (i - 1) % 2 == 0:
            x = layers.UpSampling2D(pool_size)(x)  # 14 x 14 x 64
        x = layers.BatchNormalization()(x)
        layer_neurons=layer_neurons//2

    x = layers.Conv2D(layer_neurons, filter_size, activation='relu', padding='same')(x)
    x = layers.Dropout(parameter_drop_out_percent)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(pool_size)(x)  # 14 x 14 x 64

    x = layers.Conv2D(1, filter_size, activation='sigmoid', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(parameter_drop_out_percent)(x)
    #
    # Output layer
    #
    x = layers.Reshape(input_shape)(x)  # Reshape to the original input shape

    autoencoder = models.Model(encoder_input, x)
    
    return autoencoder



# Assuming that all images have the same shape
input_shape = train_images.shape[1:]
autoencoder = build_autoencoder(input_shape, parameter_latent_dim, parameter_convolution_layers, parameter_filter_size, parameter_pool_size, parameter_drop_out_percent)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.summary()


# Configure early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.025,      # Minimum change to qualify as an improvement
    patience=5,           # Number of epochs with no improvement to stop training
    verbose=1,            # Display information about early stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)



# Fit the model with early stopping
autoencoder.fit(
    train_images, train_images,
    epochs=parameter_epochs,            # Increase the number of epochs
    batch_size=parameter_batch_size,
    shuffle=True,
    validation_data=(val_images, val_images),
    callbacks=[early_stopping]  # Use the early stopping callback
)

# Save the entire model
autoencoder.save('./giannis/autoencoder_model_3.keras')


