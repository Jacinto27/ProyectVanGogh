
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors


# input shape
input_shape = (224, 224, 3)
model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)

# Features
dataset_dir = 'Data/NormImages'
features = []
for img_path in os.listdir(dataset_dir):
    img = image.load_img(os.path.join(dataset_dir, img_path), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features.append(model.predict(x).flatten())
features = np.array(features)


k = 5
nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(features)

# Load a query image
from tensorflow.keras.applications.vgg16 import preprocess_input
query_img_path = 'Data/QueryImages/NormQueryImages/geralt drunk 02 cropped.jpg'
query_img = image.load_img(query_img_path, target_size=(224, 224))
query_img = np.expand_dims(query_img, axis=0)
query_img = preprocess_input(query_img)
query_img_feature = model.predict(query_img).flatten()


# Find the most similar images
distances, indices = nbrs.kneighbors(query_img_feature.reshape(1, -1))
for i in range(k):
    similar_img_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[indices[0][i]])
    print('Similar image {}: {}'.format(i+1, similar_img_path))


#similaridad
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image

img1 = image.load_img(os.path.join(dataset_dir, os.listdir(dataset_dir)[indices[0][0]]), target_size=(224, 224))
img2 = image.load_img(query_img_path, target_size=(224, 224))

img1_array = np.array(img1).flatten()
img2_array = np.array(img2).flatten()

# calculate the cosine similarity
similarity = cosine_similarity([img1_array], [img2_array])[0][0]

# print the cosine similarity
print('Cosine similarity:', similarity)


#TODO: incluir el cosine similarity dentro del for loop y organizar los resultaeos del print para que los incluya y los imprima en el orden del cossim
#TODO: arreglar el formato del cosine para que presente los datos en porcentaje
#TODO: ABRIR EL PRIMER PAIR DE IMAGENES CON EL MAYR COSINA
#todo: organizar los batches para que se separen por cateogira de imaenes de van gogh segun los folders pre establkecidos, por ejemplo, portraits, background, etc.
