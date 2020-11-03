from .VGG16 import VGG16
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class SSD224():

    def __init__(self):
        super(SSD224, self).__init__()

        '''
            Available layers
            Typo: layerType_Stage_NumberInStage_Info
        '''
        self.backbone = VGG16()
        self.backbone_model = self.backbone.model()

        self.flatten_1_1 = Flatten()
        self.dense_1_2_100 = Dense(100, activation='relu')
        self.dense_1_3_10 = Dense(10, activation='softmax')

    def model(self):
        '''
            Model Implementation
        '''

        model = keras.models.Sequential(
            self.backbone_model.layers[:self.backbone.getIdxFlattenLayer()])

        model.add(self.flatten_1_1)
        model.add(self.dense_1_2_100)
        model.add(self.dense_1_3_10)
        return model

    def call(self, x):
        return self.model(x)
