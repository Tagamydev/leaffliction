leaffliction
An innovative computer vision project utilizing leaf image analysis for disease recognition.

this project is made in python using pytorch to train the model and handle the data

requiriments
this project was tested using the last python 3.12.x stable version and pytorch 2.8.0 with CUDA 12.9

Install CUDA, Pytorch and Python
pytorch: https://pytorch.org/get-started/locally/
python: https://www.python.org/downloads/
CUDA: https://developer.nvidia.com/cuda-12-9-0-download-archive

install the requiriments.txt and make

make distribution - to make the analysis of the dataset
make augmentation - to augment the data in the dataset applying transformations such as: image flip, rotation, skew, shear, crop and distort
make transformation - to use methods to extract characteristics from the images
make train - to train the model based on the modified dataset
make - to do all of this in order

also you can run 
./Distriubtion.py [some dir]
./Augmentation.py [some img]
./Transformation.py -h
./train.py [dir]
./predict.py [img]
