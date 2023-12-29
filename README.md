# First Generation Pokemon Classification with a CNN
Devyn Holland <br />

## **PURPOSE** <br />
I developed a convolution neural network which can classify a first generation species of Pokemon. <br />

## **CODE WALKTHROUGH** <br />
### 1. Prepping the data: <br />
• split the data set into training and testing data via utilizing Google Drive folders (uploading all photos to colab was extensive) (I was required to use google colab as a requirement of the course I was taking) <br />
• Within the training data - create y_training data in order to validate training results <br />
• Move over random images from the training folder into the test folder <br />
• Organization: Kaggle -> kaggle.json | PokemonData -> Pokemon_test | Pokemon_train <br />
• Discovering the list of directories to confirm number of Pokemon species, as well as finding out how many photos there are of each creature <br />
• View a single creature for understanding <br />
• Incorporate data augmentations with ImageDataGenerator() - created various transformatios to exisitng training images to improve our CNN <br />
• Break out into final training and testing data (validation data) with applied data augmentation <br />
### 2. Creating the CNN: <br />
• Creating a convolutional neural network based off of the image size as well as the number of species of Pokemon within the datset <br />
• Added 7 different layers <br />
• Conv2D/ ReLU: Converts all negative values to zero while keep the positive values. This layer learns and can pick up complex patterns in the image (complex feature extraction). Removing negative values reduces computational complexity and memory requirements during training. This also solves the vanishing gradient problem during back propagation. (Non-linear) <br />
• MaxPooling2D: Selects the maximum value within each pooling window/ neighborhood. This neighborhood/window slides over the entire input image and picks up the maximum values. This helps grab the most prominent features in the image and removing the least relevant information. <br />
• Flatten: Transforms the data from multi-dimensions to a single dimensional vector, and does so without altering any values within the data. This allows the network to learn high-level features across the entire picture.
ex: (batch_size, height, width, channels) -> (batch_size, height * width * channels) <br />
• Softmax: The output layer which transforms the input scores between a range of 0 and 1. This is the layer that estimates the probabilities of each Pokemon creature. (a probability distribution) <br />
•  Model is compiled using cross-entropy loss function (commonly used with Softmax). Utilized Adam (Adaptive Moment Estimation) for the model's optimizer as it provides good generalization on datasets. This is due to Adam updating the model's weights during training to minimize the loss function which allows the CNN to make better predictions. Utilized 'categorical_crossentropy' as there are multiple species the model is trying to classify. <br />

## **SOURCES** <br />
https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files
