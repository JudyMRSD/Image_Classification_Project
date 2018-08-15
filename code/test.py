from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
import time


def testing(trained_model_path, test_dir):
    """
    Perform detection using trained keras model on color or grayscale images
    :param trained_model: keras mode *.h5 file
    :return: vector of predicted classes for test_images
    """
    start_test = time.time()
    assert (os.path.exists(trained_model_path)), "Invalid model path"
    assert (os.path.exists(test_dir)), "Invalid test image directory path"
    batch_size = 64

    model = load_model(trained_model_path)
    img_shape = model.input.shape[1:3]

    # no data augmentation (except normalize) , no shuffle for test set
    test_gen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_gen.flow_from_directory(
        test_dir,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True)
    test_loss, test_accuracy = model.evaluate_generator(test_generator)
    test_time = time.time() - start_test

    print("test accuracy = ", test_accuracy)
    print("test time = ", test_time)

def test_demo(dataset_name, model_type):
    data_dir = "../data/"
    trained_model_path = data_dir + "model/" + dataset_name + model_type + ".h5"
    test_img_dir = data_dir + dataset_name + "/test/"
    testing(trained_model_path, test_img_dir)


def main():
    dataset_name = "lung"  # use "retinal" or "lung"
    model_type = 'Resnet50'  # use lung or Resnet50
    test_demo(dataset_name, model_type)

if __name__ == '__main__':
    main()

