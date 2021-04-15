# Datamining
Datamining final project Spring 2021
Languages: R, Python?

-- Block Assignments --

Lila - 2 (Network)

Connor - 3 (Sim vector dataset)

Desmond - 4 (Sim vector comparison/output)

Michael - 1 (data ETL)

Block 1: Image Importing & Scaling
  Import dataset
  Scale images to common resolution/ratio
  Normalize pixel values
  
Block 2: Model Construction
  Import & deploy EfficientNetB7 model
  Include outputs of penultimate layer with prediction. 
  Set up tensorboard to log & visualize model.
  
Block 3: Similarity Vector Dataset
  Create dataset to store feature vectors
  loop over input dataset and feed each image through the NN.
  Organize dataset in a way that will allow for efficient searching.
  
Block 4: Similarity Metrics
  Compute the Jaccard and Cosine similarity for two N-dimensional vectors
  Search the feature vector dataset for the 3 best matches. 

If we have time:
  fine tuning model (training it on our specific subset of imagenet)
  cropping images based on bounding boxes
