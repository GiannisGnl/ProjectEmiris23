from keras.datasets import mnist
import numpy as np

def reduce_load_input_data():
    # Load input data using mnist dataset for illustration
    (train_X, _), (test_X, _) = mnist.load_data()

    # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X = np.expand_dims(test_X, axis=-1)

    # Normalize the pixel values
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    return train_X, test_X

#input_images, query_images = reduce_load_input_data()

#print('input: '  + str(input_images.shape))
#print('query: '  + str(query_images.shape))
