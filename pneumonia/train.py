from dcm2jpg import store
from model import get_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import sys


def train(epochs,w,h,steps_per_epoch,df_path,train_images):
    model=get_model((w,h,3))
    class_weight = {0: 1, 1: 3.3}

    early = EarlyStopping(monitor='accuracy', patience=3, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=2, verbose=1, cooldown=0, mode='auto',min_delta=0.0001, min_lr=0)

    store(train_images,df_path,w,h)

    datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, horizontal_flip=True,
                                 width_shift_range=0.05, rescale=1 / 255, fill_mode='nearest', height_shift_range=0.05,
                                 preprocessing_function=preprocess_input, validation_split=0.1,
                                 )

    train = datagen.flow_from_directory('data', color_mode='rgb', batch_size=128, class_mode='binary',
                                        subset='training',target_size=(w,h))
    test = datagen.flow_from_directory('data', color_mode='rgb', batch_size=32, class_mode='binary',
                                       subset='validation',target_size=(w,h))

    if steps_per_epoch:
        model.fit(train, epochs=epochs, callbacks=[early,reduce_lr], steps_per_epoch=steps_per_epoch, validation_data=test,
               class_weight=class_weight)
    else:
        model.fit(train, epochs=epochs, callbacks=[early,reduce_lr], validation_data=test,
                  class_weight=class_weight)

def main():
    my_parser = argparse.ArgumentParser(description='Pneumonia detection')

    my_parser.add_argument('--df_path',
                           metavar='df_path',
                           type=str,
                           help='the path of the dataframe containing metadata',required=True)
    
    my_parser.add_argument('--train_images_path',
                           metavar='train_images_path',
                           type=str,
                           help='the path of the directory containing training dicom files',required=True)

    my_parser.add_argument('--epochs',
                           metavar='epochs',
                           type=int,
                           help='no. of epochs',
                           action='store',
                           default=15)

    my_parser.add_argument('--steps_per_epoch',
                           metavar='steps_per_epoch',
                           type=int,
                           help='steps per epoch',
                           action='store',
                           default=0)

    my_parser.add_argument('--width',
                           metavar='width',
                           type=int,
                           help='width of image you want to resize to',
                           action='store',
                           default=256)

    my_parser.add_argument('--height',
                           metavar='height',
                           type=int,
                           help='height of image you want to resize to',
                           action='store',
                           default=256)

    args = my_parser.parse_args()
    
    train(args.epochs,args.width,args.height,args.steps_per_epoch,args.df_path,args.train_images_path)

if __name__=='__main__':
    main()
