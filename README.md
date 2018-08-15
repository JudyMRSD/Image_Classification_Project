# Project Summary
Image classification task for identifying 4 classes of retinal images, and 2 classes of lung images. </br> 



# Method
Fine tuned Resnet-50 and InceptionV3 models with pretrained weights from the convolution layers, 
and randomly initialized fc layer weights. </br>
The pretrained convolution layer weights are not frozen so all layers are trainable. </br>

# Files 
```
project
│   README.md
└───code
|   │   train.py      
|   │   test.py 
|   │   buil_model.py   Create Keras model 
│   │   utility.py      Split dataset into train, val and test
└───data
|   └───train/
│   └───val/
│   └───test/
```

# Usage

Run training on a dataset:</br> 
```
python train.py
```
Run testing on a dataset:</br> 

```
python test.py
```

For both training and testing, dataset_name can be changed to train on retina images or lung images.
model_type can be changed to train with Resnet50 or InceptionV3.


# Dataset summary
Retina image dataset, 4 classes, splitted into training, testing, validation as 8:2:2  </br>
Number of training images: 24580  </br>
Number of validation images: 3071 </br>
Number of testing images: 3077 </br>


Lung image dataset, 2 classes,  splitted into training, testing, validation as 8:2:2  </br>
Number of training images: 1396  </br>
Number of validation images: 174 </br>
Number of testing images: 177 </br>


# Result
Following are the results for fine tuning Resnet-50 or InceptionV3 models on retina or lung datasets. </br>

|                	    | Resnet-50, retina	|  Resnet-50, lung  |  InceptionV3, retina | InceptionV3, lung  | 
| ------------------    | ----------------- | ----------------- | -------------------- | ------------------ |
|Training Accuracy      | 0.938             |  0.99             |  0.84                | 0.997              | 
|Validation Accuracy (s)| 0.8857            |  0.71             |  0.799               | 0.707              | 
|Testing Accuracy (s)   | 0.8616            |  0.67             | 0.8736               | 0.644              | 
|Testing time (s)       | 44.8	            |  28.759           |  54                  | 32                 | 

# Environment
Python 3.6, Keras 2.1.3, Tensorflow 1.4

# Reference
Following are the tutorials that I found useful while doing the project.  </br>
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html</br> 
https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975</br> 
