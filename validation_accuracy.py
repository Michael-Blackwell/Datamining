from variables import *

# import labels for train & val datasets
labels_df = import_class_labels(main_dir)
class_names = labels_df.Label.tolist()

# load model
model, feat_ex = load_model(len(class_names))

# load dataset
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_path,
    image_size=(image_width, image_height),
    batch_size=1
    )

predictions = []
labels = []
for x, y in tqdm(val_data, desc="Pullin' Those Predictions! "):
    pred = model.predict(x)
    predictions = predictions + list(np.where(pred[0] == np.max(pred[0]))[0])
    labels = labels + list(y.numpy())

conf_mat = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
correct = 0
incorrect = 0

# compute total accuracy
for i in range(len(conf_mat)):
    correct += conf_mat[i][i]
    incorrect += sum(conf_mat[i]) - conf_mat[i][i]
print('--------------------------------------------------------------------------')
print('Correct Predictions: ', correct)
print('Incorrect Predictions: ', incorrect)
print('Accuracy: ', 1 - (incorrect/correct))
