import os, glob
import shutil
import csv

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np


class Tool():
    def split_data(self, split_log, train_dir, val_dir, test_dir, classes):
        '''
        When a validation dataset and test dataset doesn't exist, split data using train:test:val = 8:1:1.
        If split_log exist, use it to reproduce previous splitting result, otherwise randomly select images
        for splitting data and record the split in split_log.

        :param split_log: *.csv file record previous split of data to reproduce data splitting result
        :param train_dir: directory containing training images
        :param val_dir: directory containing validation images
        :param test_dir: directory containing testing images
        :param classes: list of class names
        '''
        assert (os.path.exists(train_dir)), "Invalid training image directory"

        train_ratio = 0.8
        val_ratio = 0.1
        # replicate previous split
        if (os.path.exists(split_log)):
            val_test_list = [x.strip() for x in open(split_log)]
        # create a new split
        # split_log first half is paths for images in training directory that would be moved to validation directory
        # split_log second half contains paths for images in training directory that would be moved to testing directory
        else:
            val_list = []
            test_list = []

            for i, cls in enumerate(classes):
                images = glob.glob(train_dir + cls + '/*.jpg')  # glob default unsorted
                num_train = int(train_ratio * len(images))
                num_val = int(val_ratio * len(images))

                # images split into    train_list   val_list   test_list
                val_list.extend(images[num_train:num_train + num_val])
                test_list.extend(images[num_train + num_val:])
            # modify paths
            for i in range(len(val_list)):
                val_list[i] = val_list[i].replace("train", "val")
            for i in range(len(test_list)):
                test_list[i] = test_list[i].replace("train", "test")

            val_test_list = val_list+test_list

            with open(split_log, 'w') as f:
                f.write("\n".join(str(item) for item in val_test_list))


        # if data is not already splitted, split data using split_log
        if (not os.path.exists(val_dir) and not os.path.exists(test_dir)):
        # create empty directories to store images
            os.makedirs(val_dir)
            os.makedirs(test_dir)
            for i, cls in enumerate(classes):
                os.makedirs(val_dir + cls + "/")
                os.makedirs(test_dir + cls + "/")

            # move images from training path to validation path and testing path
            for f in val_test_list:
                if ("val" in f):
                    source_val_path = f.replace("val", "train")
                elif ("test" in f):
                    source_val_path = f.replace("test", "train")
                # do not move if it's already a training file
                else:
                    continue
                shutil.move(src=source_val_path, dst=f)

    @staticmethod
    def visualizeTrain(history, tag, history_log_directory):
        keys_list = history.history.keys()
        print("history keys: ", keys_list)
        
        # save accuracy and loss for training and validation set
        log_list = ['val_loss', 'val_acc', 'loss', 'acc', 'lr']
        for k in keys_list:
            np.savetxt(history_log_directory + tag + k + '.txt', history.history[k])
        

        # close all previous plots
        plt.close('all')
        loss_acc_list = ['val_loss', 'val_acc', 'loss', 'acc']
        for k in loss_acc_list:
            plt.plot(history.history[k])   
        plt.legend(loss_acc_list, loc='upper right')
        history_loss_plot = history_log_directory + tag +'training_log.png'
        print("save plot of loss curve in : ", history_loss_plot)
        plt.savefig(history_loss_plot)


        # save learning rate plot
        plt.close('all')
        plt.plot(history.history['lr'])
        plt.legend(['lr'], loc='upper right')
        history_lr_plot = history_log_directory + tag +'training_learning_rate.png'
        print("save plot of learning curve in : ", history_lr_plot)
        plt.savefig(history_lr_plot)







