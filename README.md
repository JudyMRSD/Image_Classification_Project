# Project Summary
Image classification task for identifying 4 classes of retinal images, and 2 classes of lung images. </br> 
Fine tuned Resnet-50 and InceptionV3 models with pretrained weights from the convolution layers, 
and randomly initialized fc layer weights. </br>

The pretrained convolution layer weights are not frozen so all layers are trainable. </br>

Please check the following link for all the data and code:

https://drive.google.com/open?id=14e3TVHlWLSs7Tpcwmoi7SZF1BnAR93eU

# Major Updates
1. Record splitting of data as retinal.txt file to reproduce result
2. Added reduce learning rate when training loss doesn't decrease 
3. Added rotation to data augmentation and added dropout to avoid overfitting  
4. An epoch is finished after using all images in training and validation set, instead of a subset

Main observation: 
After using all images in training set with data augmentation and dropout, 
the less fluctuation on the loss curve is smaller, and the accuracy increased from
0.87 to 0.91 on testing set. 


[//]: # (Image References)
[v0_training]: ./data/writeup_images/version_0.png
[v1_training]: ./data/writeup_images/version_1.png
[v2_training]: ./data/writeup_images/version_2.png
[v2_lr]: ./data/writeup_images/version_2_lr.png

# Methods and Results   

Following are the explainations on methods and results for fine tuning InceptionV3 models on retina datasets. </br>

**Version_0** is the original version of my code that I experimented on Aug 29th. Although it has early stopping,
it finished after reaching the maxinum epochs of 100 and didn't trigger early stopping isnce validation accuracy
didn't stop improving. This is also the version that early stopped at epoch 32 when trained on Dr. Pu's machine.

Following is a table recording parameters that I changed and training results. 

For all the three versions, only the best model was saved according to some metric and used for later measurement
of training, validation and testing accuracy. For **version_0**, the metric was validation
accuracy. For **version_1** and **version_2**, the metric was changed to validation loss. This would not cause much difference 
in result, since from the final training result plots and based on how they are calculated mathmatically, 
validation accuracy becomes high when validation loss is low as they both represent the ability of the model to generalize 
on new data. 


|                	    |     version_0	    |    version_1   |        version_2        | 
| ------------------    | ----------------- | ----------------- | -------------------- | 
|Batch size             |     64	        |    64             |          64         | 
|Total Epochs Finished  |     100	        |    50             |          100         | 
|Early stopping         |metric: val_acc    |    None           |          None        |  
|Training batches per epoch |   50          |  number of samples / batch size   |number of samples / batch size  |  
|Validation batches per epoch|  10          |  number of samples / batch size   |number of samples / batch size  |  
|Dropout                |     None	        |    None	        |0.5 dropout before final layer| 
|Learning rate          |     0.0001        |0.0001, reduce when val_loss stop improving |0.0001, reduce when val_loss stop improving| 
|Data augmentation      |     Flip           |Flip               |Flip, Rotation        | 
|Training Accuracy      |     0.95	        |    0.96             |          0.96          | 
|Validation Accuracy    |     0.90	        |    0.94             |          0.94          |  
|Testing Accuracy       |     0.87	        |    0.90             |          0.91          | 
|Training time (s)      |     4172 	        |    15016             |          48984         | 
|Testing time (s)       |     51	        |    61            |          61            | 


Comparing the three training results, **version_0** has a large flucutation on validation loss. This could be explained by
it only using a subset of training data for each training epoch, and a subset of validation data for each validation epoch. 
The amount of data used was possibly not representative of the dataset, so it's also possible for the different result
on my computer and when trained on Dr. Pu's computer. 
Following is the training log for **version_0**. 

![alt text][v0_training] 

**Version_1** finishes an epoch after going through all data in the training set, thus the training loss fluctuates less.
I didn't use early stopping here, and the number of training epochs is only 50 since I was trying to validate the idea
that number of training samples involved in each epoch has a significant effect on the convergence. The learning rate
was also plotted in the image since it reduces when validation loss stops improving, 
but since the scale is relatively small compared to others, the line looks flat. 
Following is the training log for **version_1**. 

![alt text][v1_training]

**Version_2** used one more data augmentation of rotating the images in addition to flipping, 
since for retina images, since the samples could have a rotation when the image was collected. 
The validation loss was more smooth for 0th to 50th epoch compared with version_1, which
could be explained by the effect of data augmentation and dropout in avoiding overfitting. 
Following is the training log for **version_2**. 

![alt text][v2_training]

To see more clearly how the training loss changes, I plotted it on a separate graph for **version_2**. The training learning 
rate reduces to 0.1 of original learning rate when validation loss stops increasing for 10 epochs. 

![alt text][v2_lr]



# Model
Models for the three versions are accessible via:
https://drive.google.com/drive/u/0/folders/10crVNzljMUMHVogTubnlfN0aVtsaBMbH?ogsrc=32


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
|   └───retinal/
|         └───train/   before running train.py, this contains all the images 
│              └───t0/      
│              └───t1/     
│              └───t2/      
│              └──-t3/     
│         └───val/     not exist in initial data folder, will be created when run the train.py
│         └───test/    not exist in initial data folder, will be created when run the train.py
|   └───mode/          *.h5 trained model
|   └───split_log/     *.csv record how data was splitted into train, test and val
|   └───training_log/  records training history
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


# Previous Result
Following are the results for fine tuning Resnet-50 or InceptionV3 models on retina or lung datasets using version_0 code. </br>

|                	    | Resnet-50, retina	|  Resnet-50, lung  |  InceptionV3, retina | InceptionV3, lung  | 
| ------------------    | ----------------- | ----------------- | -------------------- | ------------------ |
|Training Accuracy      | 0.938             |  0.99             |  0.84                | 0.997              | 
|Validation Accuracy (s)| 0.8857            |  0.71             |  0.799               | 0.707              | 
|Testing Accuracy (s)   | 0.8616            |  0.67             | 0.8736               | 0.644              | 
|Testing time (s)       | 44.8	            |  28.759           |  54                  | 32                 | 

# Environment
Python 3.6, Keras 2.1.3, Tensorflow 1.4
Training used Titan X Pascal 12 GB 
# Reference
Following are the tutorials that I found useful while doing the project.  </br>
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html</br> 
https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975</br> 
