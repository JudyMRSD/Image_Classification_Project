import os, glob
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np



class Tool():
    def split_data(self, train_dir, val_dir, test_dir, classes):
        # since a validation dataset and test dataset doesn't exist, split data using train:test:val = 8:1:1
        assert (os.path.exists(train_dir)), "Invalid training image directory"

        train_ratio = 0.8
        val_ratio = 0.1

        if (not os.path.exists(val_dir) and not os.path.exists(test_dir)):
            os.makedirs(val_dir)
            os.makedirs(test_dir)
            for i, cls in enumerate(classes):
                os.makedirs(val_dir + cls + "/")
                os.makedirs(test_dir + cls + "/")

                all_imgs = glob.glob(train_dir + cls + '/*.jpg')

                num_train = int(train_ratio * len(all_imgs))
                num_val = int(val_ratio * len(all_imgs))

                # images split into    train_list   val_list   test_list
                val_list = all_imgs[num_train:num_train+num_val]
                test_list = all_imgs[num_train+num_val:]

                for f in val_list:
                    val_path = f.replace("train", "val")
                    shutil.move(src=f, dst=val_path)
                for f in test_list:
                    test_path = f.replace("train", "test")
                    shutil.move(src=f, dst=test_path)
    @staticmethod
    def visualizeTrain(history, tag, history_log_directory):
        keys_list = history.history.keys()
        print("history keys: ", keys_list)
        # close all previous plots
        plt.close('all')
        # save accuracy and loss for training and validation set
        for k in keys_list:
            np.savetxt(history_log_directory + tag + k + '.txt', history.history[k])
            plt.plot(history.history[k])
            
        plt.legend(keys_list, loc='upper right')
        history_loss_plot = history_log_directory + tag +'training_log.png'
        print("save plot of loss curve in : ", history_loss_plot)
        plt.savefig(history_loss_plot)






