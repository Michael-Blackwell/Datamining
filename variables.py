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
import re

########################
# ---- Parameters ---- #
########################
top_images = 3
image_width, image_height = 260, 260
batch_size = 1

############################
# ---- Path Variables ---- #
############################
main_dir = '/home/mike/Public/School/datamining/Project'

test_img_dir = main_dir + '/test'
train_data_path = main_dir + '/imagenet-mini/train'
val_data_path = main_dir + '/imagenet-mini/val'
pickling_path = main_dir + '/pickles/features_df'

#######################
# ---- Functions ---- #
#######################
def load_model(classes):
    # Load the pre-trained model
    model = tf.keras.applications.EfficientNetB2(
        weights="imagenet",
        classes=classes
    )
    # Create object to extract image features from penultimate layer
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    return model, feat_extractor


def import_class_labels(main_directory):
    # import labels from disk into a dataframe
    labels = pd.read_json(main_directory + '/imagenet-simple-labels.json')
    labels.rename(columns={0: 'Label'}, inplace=True)

    # list of folder labels from dataset
    files = os.listdir(main_directory + '/imagenet-mini/train')
    files.sort()
    # sorting the class folders ensures they are in the same order as the labels list
    file_names = pd.DataFrame({'filename': files})
    # join the two df's together so we have a mapping of folder name to class name.
    labels = labels.join(file_names)
    labels = labels.set_index('filename')
    return labels


def create_dataset(path, img_width, img_height):
    # Create a dataset from images on disk
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(img_width, img_height)
        # batch_size=batch
    )
    return dataset


def import_gen(image_name_map, f_extractor):
    # generator object for importing raw images
    for _ in tqdm(image_name_map, desc='Importing Images'):
        original_img = load_img(_[1], target_size=(image_width, image_height))
        img_array = img_to_array(original_img)
        reshaped_img_array = np.expand_dims(img_array, axis=0)
        normalized_img_array = preprocess_input(reshaped_img_array)
        yield _, f_extractor.predict(normalized_img_array)


def import_images(path, feature_extractor, labels):
    # load raw images from disk & return features dataframe
    image_name_map = []
    image_class_map = []
    # Create a list of images to be imported (image paths on disk)
    for root, dirs, files in tqdm(os.walk(path), desc='Mapping Images'):
        for name in files:
            path = os.path.join(root, name)
            image_name_map.append(path)
            image_class_map.append(os.path.basename(os.path.dirname(path)))
    image_class_map = list(map(lambda x: labels.loc[x].Label, image_class_map))
    multi_index_tuple = list(zip(image_class_map, image_name_map))
    multi_index = pd.MultiIndex.from_tuples(multi_index_tuple, names=('label', 'path'))

    # Create empty dataframe for features to be pasted into
    features_df = pd.DataFrame(columns=np.arange(feature_extractor.output.shape[1]), index=multi_index)
    # Use generator object 'import_gen' to iterate over list of image paths, extract feats, and paste into df
    # (approx 30 min with i7 6600k 4GHz)

    for idx, features in tqdm(import_gen(multi_index_tuple, feature_extractor), desc='Importing Images'):
        features_df.loc[idx] = features.tolist()[0]

    return features_df


def cosine_sim(a, b): return dot(a, b)/(norm(a)*norm(b))
# Computes cosine similarity score of two numpy arrays


def compare_similarity(test_features, test_prediction, features_df):
    # Identifies and returns the dataset images most similar to the test image
    # Compute Cosine Sim
    cos_similarities_df = pd.DataFrame(columns=['Cos Sim'])
    test_feat_indx = list(test_features.index)[0]
    for label, path in tqdm(features_df.index, desc="Computing Cosine Similarity:"):
        if label == test_prediction:
            cos_similarities_df.loc[path] = cosine_sim(features_df.loc[(label, path)],
                                                       test_features.loc[test_feat_indx])

    cos_results = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[1:top_images + 1].index

    # Show results
    plots = int((top_images+1)/2) + 1
    rows = 2
    figure, axis = plt.subplots(rows, plots)

    # plot test image
    original = load_img(list(test_features.index)[0], target_size=(image_width, image_height))
    axis[0, 1].imshow(original)
    axis[0, 1].set_title('Test Image')

    # plot similar images
    closest_imgs = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[0:top_images].index
    closest_imgs_scores = cos_similarities_df['Cos Sim'].sort_values(ascending=False)[0:top_images]

    for i in range(len(closest_imgs)):
        actual_path = re.sub('/home/mike/Public/School/datamining/Project', main_dir, closest_imgs[i])
        img = load_img(actual_path, target_size=(image_width, image_height))
        axis[int(i/plots)+1, (i % plots)].imshow(img)
        axis[int(i/plots)+1, (i % plots)].set_title(closest_imgs_scores[i])

    plt.show()
