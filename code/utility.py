import os, glob
import shutil

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






