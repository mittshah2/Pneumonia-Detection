import os
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm
import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt

def store(train_images,df_path,w=256,h=256):

    if os.path.exists('data')==False:
        os.mkdir('data')
        os.mkdir(os.path.join('data','positive'))
        os.mkdir(os.path.join('data','negative'))

    df = pd.read_csv(df_path)
    x=np.asarray(df['patientId'].astype(str) + '.dcm')

    for i in range(len(x)):
        x[i]=os.path.join(train_images,x[i])

    df['path'] = x

    negative = df[df['Target'] == 0]

    positive = df[df['Target'] == 1]
    unique_positive = positive[['path', 'patientId']]
    path = unique_positive['path'].unique()
    patientId = unique_positive['patientId'].unique()

    unique_positive = pd.DataFrame({'path': path, 'patientId': patientId})
   

    for _, row in tqdm(unique_positive.iterrows()):
        img = dicom.read_file(row['path']).pixel_array
        img = resize(img, (w, h))
        plt.imsave(os.path.join(os.path.join('data','positive') ,str(row['patientId']) + '.jpg' ), img, cmap='gray')

    for _, row in tqdm(negative.iterrows()):
        img = dicom.read_file(row['path']).pixel_array
        img = resize(img, (w, h))
        plt.imsave(os.path.join(os.path.join('data','negative') , str(row['patientId']) + '.jpg'), img, cmap='gray')





