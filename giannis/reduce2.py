import argparse
import numpy as np
#from testforinput2 import load_input_data
#from testforinput2 import reduce_load_input_data
from keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras import models
import math

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

#Load input data
#def reduce_load_input_data(dataset_path,query_path): prepei na pairnei ta 2 auta orismata apo cmd line kai na ta kanei kapws load stis train_X kai test_X
def reduce_load_input_data():
    # Load input data using mnist dataset for illustration
    (train_X, _), (test_X, _) = mnist.load_data()
    print("Shape of dataset:", train_X.shape)
    print("Shape of dataset:", test_X.shape)
    # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)

    # Normalize the pixel values
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    return train_X, test_X




# Load the entire model
loaded_autoencoder = load_model('autoencoder_model.keras')

# Extract the encoder part
loaded_encoder = models.Model(inputs=loaded_autoencoder.input, outputs=loaded_autoencoder.layers[-18].output)

# Load input data using the function from testforinput2
input_images, val_images = reduce_load_input_data()
 

# Use the loaded encoder for predictions
compressed_dataset = loaded_encoder.predict(input_images)
compressed_queryset= loaded_encoder.predict(val_images)



print("Shape of compressed_dataset:", compressed_dataset.shape)


def find_max_value(compressed_dataset):
    # Flatten the compressed_dataset to get a 2D array (60,000, 15)
    flattened_dataset = compressed_dataset.reshape((compressed_dataset.shape[0], -1))

    # Find the maximum value across all elements
    max_value = np.max(flattened_dataset)
    print(max_value)
    return max_value





#Finding the max values of compressed datasets
max_value= find_max_value(compressed_dataset) #gia dataset
max_value_queryset= find_max_value(compressed_queryset) #gia queryset


#function to normalize number
def normalize_num(number,max_value,max_int): #normalize 1 number each time
    r=max_value/max_int
    normal_number =math.ceil(number/r)
    return normal_number


# Function to normalize the entire compressed dataset
def normalize_dataset(compressed_dataset, max_value, max_int):
    normalized_dataset = np.vectorize(lambda x: normalize_num(x, max_value, max_int))(compressed_dataset)
    return normalized_dataset

max_int = 255

# Normalize the compressed dataset using the normalize_dataset function
normalized_dataset = normalize_dataset(compressed_dataset, max_value, max_int)
normalized_queryset = normalize_dataset(compressed_queryset, max_value_queryset, max_int)


# Save the compressed dataset for both output files
np.save(args.output_dataset_file, normalized_dataset)
np.save(args.output_query_file, normalized_queryset)
