# 1st Test for results (functional) Optimals so far:epochs=4, batchsize=512

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from sklearn.model_selection import train_test_split
import argparse
import os
from testforinput2 import load_input_data  # Import the correct function

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train an image autoencoder.')
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='Path to the input dataset directory with images.')
parser.add_argument('-q', '--queryset', type=str, required=True,
                    help='Path to the input query set directory with images.')
parser.add_argument('-od', '--output_dataset_file', type=str, required=True,
                    help='Path to the output reduced dataset file.')
parser.add_argument('-oq', '--output_query_file', type=str, required=True,
                    help='Path to the output reduced query file.')
args = parser.parse_args()

# Load input data using the function from testforinput2
input_images, input_labels = load_input_data()  # Call the correct function

# Split into train and validation sets
train_images, val_images = train_test_split(input_images, test_size=0.1, random_state=42)

# Autoencoder architecture
latent_dim = 10  # default compression dimension, you could use it as a CLI argument

#1o set hyperparameters

def build_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    decoder_input = layers.Dense(np.prod(x.shape[1:]), activation='relu')(latent_vector)
    x = layers.Reshape((7, 7, 64))(decoder_input)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    encoder = models.Model(encoder_input, latent_vector)
    autoencoder = models.Model(encoder_input, decoder_output)
    
    return autoencoder, encoder

#2o try me set hyperparameters ----> KAKO!! more time to run and bigger loss
""""
def build_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    decoder_input = layers.Dense(7 * 7 * 64, activation='relu')(latent_vector)  # Manually specify the size
    x = layers.Reshape((7, 7, 64))(decoder_input)  # Adjusted reshape
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    encoder = models.Model(encoder_input, latent_vector)
    autoencoder = models.Model(encoder_input, decoder_output)
    
    return autoencoder, encoder
"""

#3o try me set hyperparameters ----> KAKO pio argo kai less expressive
"""
def build_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    decoder_input = layers.Dense(7 * 7 * 64, activation='relu')(latent_vector)  # Manually specify the size
    x = layers.Reshape((7, 7, 64))(decoder_input)  # Adjusted reshape
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    encoder = models.Model(encoder_input, latent_vector)
    autoencoder = models.Model(encoder_input, decoder_output)
    
    return autoencoder, encoder
"""




# Assuming that all images have the same shape
input_shape = train_images.shape[1:]
autoencoder, encoder = build_autoencoder(input_shape, latent_dim)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_images, train_images,
                epochs=4,
                batch_size=512,
                shuffle=True,
                validation_data=(val_images, val_images))

# Function to save the compressed images
def save_compressed_images(images, encoder, file_path):
    compressed = encoder.predict(images)
    np.save(file_path, compressed)

# Saving output compressed files
save_compressed_images(input_images, encoder, args.output_dataset_file)
save_compressed_images(input_images, encoder, args.output_query_file)




#try gia visualization!

"""
# Load compressed representations from output files
compressed_dataset = np.load('output_dataset_file.npy')
compressed_query = np.load('output_query_file.npy')

# Assuming you have a function to create the autoencoder model
# (autoencoder, encoder) = build_autoencoder(input_shape, latent_dim)

# Decode compressed representations to get reconstructed images
decoded_dataset = autoencoder.predict(compressed_dataset)
decoded_query = autoencoder.predict(compressed_query)

# Function to display original and reconstructed images side by side
def display_comparison(original, reconstructed, title):
    n = min(len(original), 9)  # Display up to 9 images
    fig, axes = plt.subplots(2, n, figsize=(15, 6))
    fig.suptitle(title)

    for i in range(n):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.show()

# Display comparison for dataset
display_comparison(train_images[:9], decoded_dataset[:9], 'Dataset: Original vs. Reconstructed')

# Display comparison for query set
display_comparison(val_images[:9], decoded_query[:9], 'Query Set: Original vs. Reconstructed')
"""