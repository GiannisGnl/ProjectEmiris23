import argparse
import idx2numpy
import numpy as np
from keras.datasets import mnist

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

# Parse the command line arguments
args = parser.parse_args()

#Load input data
def reduce_load_input_data():
    # Load input data using mnist dataset for illustration
    # array = idx2numpy.convert_from_file(args.dataset)
    # (train_X, _), (test_X, _) = idx2numpy.convert_from_file(args.dataset), idx2numpy.convert_from_file(args.queryset)
    train_X = idx2numpy.convert_from_file(args.dataset)
    test_X = idx2numpy.convert_from_file(args.queryset)
    
     # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)

    # Normalize the pixel values
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    print(len(train_X))
    print(len(test_X))


#Load input data
def reduce_load_input_data_test():
    # Load input data using mnist dataset for illustration
    (train_X, _), (test_X, _) = mnist.load_data()
    print(len(mnist.load_data()))
    # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)

    # Normalize the pixel values
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # return train_X, test_X


# Access and print the values of the arguments
# print('Argument A:', args.dataset)
# print('Argument B:', args.argument_b)

# test = reduce_load_input_data_test()
test = reduce_load_input_data()