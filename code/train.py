from utility import Tool
from build_model import Finetune_Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import time

def training(model, training_img_dir, validation_img_dir, trained_model_path):
    start_train = time.time()
    assert (os.path.exists(training_img_dir)), "Invalid training image directory"
    assert (os.path.exists(validation_img_dir)), "Invalid validation image directory"

    batch_size = 64
    img_shape = model.input.shape[1:3]

    # data augmentation: flip images
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_gen.flow_from_directory(
        training_img_dir,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True)

    # no data augmentation (except normalize) , no shuffle for validation set
    val_gen = ImageDataGenerator(rescale=1. / 255)


    val_generator = val_gen.flow_from_directory(
        validation_img_dir,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True)
     
    # Early stopping and save the best model according to accuracy on the validation set
    callbacks = [EarlyStopping(monitor='val_acc', patience=20),
                 ModelCheckpoint(filepath=trained_model_path, monitor='val_acc', save_best_only=True)]
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(train_generator,
                        epochs=100,
                        steps_per_epoch=50,
                        validation_data=val_generator,
                        validation_steps=10,
                        callbacks=callbacks,
                        verbose=1)
    train_time = time.time() - start_train
    train_loss, train_accuracy = model.evaluate_generator(train_generator)
    val_loss, val_accuracy = model.evaluate_generator(val_generator)

    print("train time = ", train_time)
    print("train_accuracy", train_accuracy)
    print("val_accuracy", val_accuracy)

def train_demo(dataset_name, model_type):
    data_dir = "../data/"
    if dataset_name == "retinal":
        num_classes = 4
        class_names = ['t0', 't1', 't2', 't3']
    elif dataset_name == "lung":
        num_classes = 2
        class_names = ['N', 'P']
    else:
        print("Error: invalid model type")
        return
    training_img_dir = data_dir + dataset_name + "/train/"
    validation_img_dir = data_dir + dataset_name + "/val/"
    test_img_dir = data_dir + dataset_name + "/test/"
    trained_model_path = data_dir + "model/" + dataset_name + model_type + ".h5"

    fineTuneModel = Finetune_Model()
    tool = Tool()
    print("training_img_dir", training_img_dir)
    tool.split_data(train_dir=training_img_dir, val_dir=validation_img_dir,
                    test_dir=test_img_dir, classes=class_names)

    fineTuneModel.build_model(model_type, num_classes)

    training(model=fineTuneModel.model, training_img_dir=training_img_dir, validation_img_dir=validation_img_dir,
            trained_model_path=trained_model_path)


def main():
    dataset_name = "lung"  # use "retinal" or "lung"
    model_type = 'InceptionV3'  # use InceptionV3,  or Resnet50
    train_demo(dataset_name, model_type)


if __name__ == '__main__':
    main()

