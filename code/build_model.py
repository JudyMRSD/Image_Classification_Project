from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras import optimizers
from keras.layers.core import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3


class Finetune_Model():
    def __init__(self):
        self.model = None
        self.model_type = None
    
    def build_model(self, model_type, num_classes):
        """
        Create model for image classification task
        :param model_type: choose between Resnet50 and InceptionV3
        :param input_image_shape: color (w x h x channels=3)
        :param num_classes: shape of output layer of the classification model
        :return: model
        """
        self.model_type = model_type
        if self.model_type == "Resnet50":
            self.build_Resnet50(num_classes)
        elif self.model_type == "InceptionV3":
            self.build_InceptionV3(num_classes)
        else:
            raise ValueError('Invalid model type')

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=0.0001),
                      metrics=['accuracy'])

    def build_InceptionV3(self, num_classes):
        # Keras vgg16 model is pretrained on images of shape (299,299,3)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        # add dropout to avoid overfitting 
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        self.model = Model(input=base_model.input, output=predictions)

    def build_Resnet50(self, num_classes):
        # Keras Resnet50 model is pretrained on images of shape (224,224,3)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_top = base_model.output
        fc0 = Flatten()(resnet_top)
        prediction = Dense(num_classes, activation='softmax')(fc0)
        self.model = Model(inputs = base_model.input, outputs=prediction)
