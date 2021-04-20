from variables import *


# import images as datasets
test_path = main_dir + '/imagenet-mini/temp'

# import labels for train & val datasets
labels_df = import_data(main_dir)
class_names = labels_df.Label.tolist()

# initiate model
model, feat_extractor = load_model(numclasses)

# import image dataset from disk, preserving image path names
image_names, main_ds = import_images(val_data_path)

# extract the images features
imgs_features = feat_extractor.predict(main_ds)  # TODO make sure order of images is retained
del main_ds  # delete after features are extracted to save space.
print("features extracted!")

features_df = pd.DataFrame(imgs_features, index=image_names)

predict(test_img_dir, feat_extractor, features_df)
