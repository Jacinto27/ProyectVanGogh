# This is a sample Python script.
import os


# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


### model 1
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Define a function to extract features from images
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Extract features from two images
img1_path = 'path/to/image1.jpg'
img2_path = 'path/to/image2.jpg'
features1 = extract_features(img1_path, model)
features2 = extract_features(img2_path, model)

# Compute cosine similarity between the two images
similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
print('Cosine similarity:', similarity)

####model 2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Extract features from images in the dataset
dataset_dir = 'Data/Vangogh/RootData'
features = []
for img_path in os.listdir(dataset_dir):
    img = image.load_img(os.path.join(dataset_dir, img_path), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features.append(model.predict(x).flatten())
features = np.array(features)

# Build a k-NN model
k = 5
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(features)

# Load a query image
query_img_path = 'path/to/query_image.jpg'
query_img = image.load_img(query_img_path, target_size=(224, 224))
query_img_feature = model.predict(image.img_to_array(query_img)).flatten()

# Find the most similar images
distances, indices = nbrs.kneighbors(query_img_feature.reshape(1, -1))
for i in range(k):
    similar_img_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[indices[0][i]])
    print('Similar image {}: {}'.format(i+1, similar_img_path))
