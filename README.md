# Plant-recognition

## Dataset
Download one of the plant datasets from the clef page.

e.g. https://www.imageclef.org/2013/plant

## Dataset preparation
The `im2list.py` script will go through the dataset folders and create a list containing labels and image paths for training data (80%) and validation data(20%).
 
option | help
------------ | -------------
root_dir | Path prefix. Default is the current directory
train_paths | Paths to training data. Multiple paths separated by space

