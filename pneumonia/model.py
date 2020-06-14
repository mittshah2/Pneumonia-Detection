from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization,LeakyReLU,GaussianDropout
from tensorflow.keras.models import Model

def get_model(input_shape=(256,256,3)):
    pre_trained_model = VGG19(input_shape=input_shape,
                              include_top=False,
                              weights='imagenet')

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output


    model = Flatten()(last_output)
    model = Dense(1024)(model)
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.25)(model)
    model = BatchNormalization()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.25)(model)
    model = BatchNormalization()(model)
    model = Dense(1, activation='sigmoid')(model)

    fmodel = Model(pre_trained_model.input, model)

    fmodel.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return fmodel