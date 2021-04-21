from variables import *

# import labels for train & val datasets
labels_df = import_class_labels(main_dir)
class_names = labels_df.Label.tolist()

# initiate model
model, feat_extractor = load_model(len(class_names))

# import dataset from disk and extract image features.
features_df = import_images(train_data_path, feat_extractor)
print("features extracted!")

# Pickle features df to disk so extraction is not necessary each time.
features_df.to_pickle(pickling_path)
