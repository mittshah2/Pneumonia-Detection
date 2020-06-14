import numpy as np
import pandas as pd
import shutil,os,argparse,sys
from tqdm import tqdm
import pydicom as dicom
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def generate_samples(train_images,df_path,size):

    os.chdir('../pneumonia/data')

    files=os.listdir('positive')

    files=np.random.choice(files,size=size)

    os.chdir('../../pneumonia_yolo')
    os.mkdir('train_images')

    df=pd.read_csv(df_path)

    for i in tqdm(files):
        name=i[:-4]+'.txt'
        path=os.path.join('train_images',name)

        img_dcm_path=os.path.join(train_images,i[:-4]+'.dcm')
        img=dicom.read_file(img_dcm_path).pixel_array
        plt.imsave(os.path.join('train_images',i),img,cmap='gray')


        fil = open(path, 'a')
        s = df[df['patientId'] == i[:-4]]
        for _,row in s.iterrows():
          st=str(0)+' '+str(row['x']/1024)+' '+str(row['y']/1024)+' '+str(row['width']/1024)+' '+str(row['height']/1024)+ '\n'
          fil.write(st)
        fil.close()

    image_files = []
    path = 'train_images'
    base=os.getcwd()
    os.chdir(path)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append(os.path.join(base,os.path.join(path , filename)))
    os.chdir("..")
    with open("train.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    os.chdir("..")

my_parser = argparse.ArgumentParser(description='yolo')

my_parser.add_argument('--df_path',
                       metavar='df_path',
                       type=str,
                       help='the path of the dataframe containing metadata', required=True)

my_parser.add_argument('--train_images_path',
                       metavar='train_images_path',
                       type=str,
                       help='the path of the directory containing training dicom files', required=True)

my_parser.add_argument('--size',
                       metavar='size',
                       type=int,
                       help='no. of examples to train the yolo',
                       action='store',
                       default=1200)
args = my_parser.parse_args()

generate_samples(args.train_images_path,args.df_path,args.size)

