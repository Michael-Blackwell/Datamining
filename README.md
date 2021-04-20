# Datamining

Datamining final project Spring 2021

Languages: Python

Create the following directory hierarchy for the program files:

Project
|_imagenet-mini (main dataset extracted)
|_test
  |_imgs
  
test images (one at a time) are placed in the imgs folder. The imagenet-simple-labels.json file is placed in the Project folder. 
after running 'main', you can change the test picture in the imgs folder and rerun predict() without reimporting the entire validation dataset again.

The method I am using to import the pictures is highly inefficient and uses a boatload of memory. If we had more time we could fix this, but right now we just have to use a subset of the whole dataset. Currently the script is using the validation dataset (approximately 3900 images as opposed to 39000) and it will use around 7 gigs of ram. If you have less than 12-16 gigs of memory it may crash your system, so be aware of that. 
