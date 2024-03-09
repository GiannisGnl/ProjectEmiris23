# 1st Test for Input function

import numpy as np

from keras.datasets import mnist

def load_input_data():
    # Load input data using mnist dataset for illustration
    from keras.datasets import mnist
    (train_X, _), (_, _) = mnist.load_data()
    
    # Add a channel dimension to simulate grayscale images
    train_X = np.expand_dims(train_X, axis=-1)
    
    # Normalize the pixel values
    train_X = train_X / 255.0
    
    return train_X, None  # Assuming there are no labels for this example


"""
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
"""



def display_images(images):
    from matplotlib import pyplot

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(images[i], cmap=pyplot.get_cmap('gray'))

    pyplot.show()

if __name__ == "__main__":
    input_images, input_labels = load_input_data()
    print('X_train: ' + str(input_images.shape))
    #print('Y_train: ' + str(input_labels.shape))

    # Display the first 9 images
    display_images(input_images[:9])



