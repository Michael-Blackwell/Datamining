import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm
import pathlib

########################
# ---- Parameters ---- #
########################
top_images = 3
image_width, image_height = 260, 260
batch_size = 1
numclasses = 1000

############################
# ---- Path Variables ---- #
############################
main_dir = '/home/mike/Public/School/datamining/Project'
test_img_dir = main_dir + '/test'
train_data_path = main_dir + '/imagenet-mini/train'
val_data_path = main_dir + '/imagenet-mini/val'

#######################
# ---- Functions ---- #
#######################
def load_model(classes):
    # Load the pre-trained model
    model = tf.keras.applications.EfficientNetB2(
        weights="imagenet",
        classes=classes
    )
    # Get image features from penultimate layer
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    return model, feat_extractor


def import_data(main_directory):
    # import labels
    labels = pd.read_json(main_directory + '/imagenet-simple-labels.json')
    labels.rename(columns={0: 'Label'}, inplace=True)

    # list of folder labels from dataset
    files = os.listdir(main_directory + '/imagenet-mini/train')
    files.sort()
    file_names = pd.DataFrame({'filename': files})
    labels = labels.join(file_names)
    return labels


def create_dataset(path, img_width, img_height, batch):
    # Create a dataset from images on disk
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(img_width, img_height),
        batch_size=batch
    )
    return dataset


def import_images(path):
    # load images from disk into a dataset
    image_name_map = {}
    for root, dirs, files in os.walk(path):
        for name in tqdm(files, desc='Mapping Images'):
            image_name_map[os.path.join(root, name)] = name
    importedimages = []
    count = 0
    for image in tqdm(image_name_map, desc='Importing Images'):
        count += 1
        original = load_img(image, target_size=(image_width, image_height))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        importedimages.append(image_batch)
    images = np.vstack(importedimages)
    processed_imgs = preprocess_input(images.copy())
    return image_name_map, processed_imgs


def cosine_sim(a, b): return dot(a, b)/(norm(a)*norm(b))
# Computes cosine similarity score of two numpy arrays


def jaccard_sim(a, b): return
# Computes jaccard similarity score of two numpy arrays


def predict(test_img_path, feature_extractor, features_df):
    # Identifies and returns the dataset images most similar to the test image
    # import test image
    test_data = pathlib.Path(test_img_path)
    test_image_names, test_ds = import_images(test_data)

    # get features of test image
    test_image_features = feature_extractor.predict(test_ds)

    # Compute Similarity Indicies

    # Cosine Sim
    cos_similarities_df = pd.DataFrame(columns=['Cos Sim'])
    for index in features_df.index:
        cos_similarities_df.loc[index] = cosine_sim(features_df.loc[index], test_image_features[0])

    cos_results = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[1:top_images + 1].index

    # Show results
    plots = int((top_images+1)/2) + 1
    rows = 2
    figure, axis = plt.subplots(rows, plots)

    # plot test image
    original = load_img(list(test_image_names.keys())[0], target_size=(image_width, image_height))
    axis[0, 1].imshow(original)
    axis[0, 1].set_title('Test Image')

    # plot similar images
    closest_imgs = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[0:top_images].index
    closest_imgs_scores = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[0:top_images]

    for i in range(len(closest_imgs)):
        img = load_img(closest_imgs[i], target_size=(image_width, image_height))
        axis[int(i/plots)+1, (i % plots)].imshow(img)
        axis[int(i/plots)+1, (i % plots)].set_title(closest_imgs_scores[i])

    plt.show()
