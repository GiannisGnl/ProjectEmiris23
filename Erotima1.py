import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from sklearn.model_selection import train_test_split
import argparse
import os
from testforinput2 import load_input_data  # Import the function from testforinput2



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

# Load images and preprocess
def load_images(image_dir):
    image_list = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize to [0, 1]
        image_list.append(img)
    return np.asarray(image_list)

dataset = load_images(args.dataset)
queryset = load_images(args.queryset)

# Split into train and validation sets
train_images, val_images = train_test_split(dataset, test_size=0.1, random_state=42)

# Autoencoder architecture
latent_dim = 10  # default compression dimension, you could use it as a CLI argument

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
    x = layers.Reshape((x.shape[1], x.shape[2], x.shape[3]))(decoder_input)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    encoder = models.Model(encoder_input, latent_vector)
    autoencoder = models.Model(encoder_input, decoder_output)
    
    return autoencoder, encoder

# Assuming that all images have the same shape
input_shape = train_images.shape[1:]
autoencoder, encoder = build_autoencoder(input_shape, latent_dim)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_images, train_images,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(val_images, val_images))

# Function to save the compressed images
def save_compressed_images(images, encoder, file_path):
    compressed = encoder.predict(images)
    np.save(file_path, compressed)

# Saving output compressed files
save_compressed_images(dataset, encoder, args.output_dataset_file)
save_compressed_images(queryset, encoder, args.output_query_file)

"""
To run the script, use the command line as follows:


shell
$ python reduce.py -d /path/to/dataset -q /path/to/queryset -od /path/to/output_dataset_file.npy -oq /path/to/output_query_file.npy

Replace `/path/to/dataset`, `/path/to/queryset`, `/path/to/output_dataset_file`, and `/path/to/output_query_file` with the actual paths you want to use.

Remember, this code is a starting framework. Ideally, you need to include various validations, like checking the existence of directories, adjusting image sizes, trying out different architectures, hyperparameters, and tuning the model based on the results of your experiments. This will allow you to reach optimal compression while minimizing error and avoiding overfitting.
"""


