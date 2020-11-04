from .VGG16 import VGG16
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


class SSD300():

    def __init__(self):
        super(SSD300, self).__init__()

        '''
            Available layers
            Typo: layerType_Stage_NumberInStage_Info
        '''
        self.backbone = VGG16(input_shape=(300, 300, 3))
        self.backbone_model = self.backbone.getModel()

        self.flatten_1_1 = Flatten()
        self.dense_1_2_100 = Dense(100, activation='relu')
        self.dense_1_3_10 = Dense(10, activation='softmax')

        '''
            Model Implementation
        '''
        self.model = keras.models.Sequential(
            self.backbone_model.layers[:self.backbone.getIdxFlattenLayer()])

        self.model.add(self.flatten_1_1)
        self.model.add(self.dense_1_2_100)
        self.model.add(self.dense_1_3_10)

    def getModel(self):
        return self.model

    def call(self, x):
        return self.model(x)
