#The better test (Version 2)~(epochs=75,batch_size=512)+(Earlystopping: monitor='val_loss',min_delta=0.01,patience=5)+(latent_dimension=15)
"""

run command: python test2.py -d /path/to/dataset -q /path/to/queryset -od c:/Users/Giannaras/Desktop/D.I.T/Project_Emiris_2023/project3/output_dataset_file.npy 
-oq c:/Users/Giannaras/Desktop/D.I.T/Project_Emiris_2023/project3/output_query_file.npy


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks
from sklearn.model_selection import train_test_split
import argparse
from testforinput2 import load_input_data  # Import the correct function

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


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
latent_dim = 15  # default compression dimension, you could use it as a CLI argument

#1o try autoencoder

"""
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
"""

#2o try autoencoder, more convolutional layers
    
def build_autoencoder(input_shape, latent_dim):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    decoder_input = layers.Dense(np.prod(input_shape), activation='relu')(latent_vector)
    x = layers.Reshape(input_shape)(decoder_input)  # Reshape to the original input shape
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    
    encoder = models.Model(encoder_input, latent_vector)
    autoencoder = models.Model(encoder_input, decoder_output)
    
    return autoencoder, encoder



# Assuming that all images have the same shape
input_shape = train_images.shape[1:]
autoencoder, encoder = build_autoencoder(input_shape, latent_dim)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Configure early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    min_delta=0.01,      # Minimum change to qualify as an improvement
    patience=5,           # Number of epochs with no improvement to stop training
    verbose=1,            # Display information about early stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Fit the model with early stopping
autoencoder.fit(
    train_images, train_images,
    epochs=75,            # Increase the number of epochs
    batch_size=512,
    shuffle=True,
    validation_data=(val_images, val_images),
    callbacks=[early_stopping]  # Use the early stopping callback
)

# Function to save the compressed images
def save_compressed_images(images, encoder, file_path):
    compressed = encoder.predict(images)
    np.save(file_path, compressed)

# Saving output compressed files
save_compressed_images(input_images, encoder, args.output_dataset_file)
save_compressed_images(input_images, encoder, args.output_query_file)


# Save the entire model
autoencoder.save('autoencoder_model.h5')


# Load the entire model
loaded_autoencoder = load_model('autoencoder_model.h5')

# Extract the encoder part (assuming encoder is the second-to-last layer)
loaded_encoder = models.Model(inputs=loaded_autoencoder.input, outputs=loaded_autoencoder.layers[-2].output)

# Use the loaded encoder for predictions
compressed_dataset = loaded_encoder.predict(input_images)




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