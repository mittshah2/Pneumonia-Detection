import os
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm
import pydicom as dicom

def store(df_path,w=256,h=256):

    os.mkdir('data')
    os.mkdir('data/positive')
    os.mkdir('data/negative')
    df = pd.read_csv(df_path)

    df['path'] = df_path + df['patientId'].astype(str) + '.dcm'

    negative = df[df['Target'] == 0]

    positive = df[df['Target'] == 1]
    unique_positive = positive[['path', 'patientId']]
    path = unique_positive['path'].unique()
    patientId = unique_positive['patientId'].unique()

    unique_positive = pd.DataFrame({'path': path, 'patientId': patientId})

    current_dir=os.getcwd()
    os.chdir(current_dir)
    print('cur--',current_dir)

    for _, row in tqdm(unique_positive.iterrows()):
        img = dicom.read_file(row['path']).pixel_array
        img = resize(img, (w, h))
        plt.imsave('data/positive/' + row['patientId'] + '.jpg', img, cmap='gray')

    for _, row in tqdm(negative.iterrows()):
        img = dicom.read_file(row['path']).pixel_array
        img = resize(img, (w, h))
        plt.imsave('data/negative/' + row['patientId'] + '.jpg', img, cmap='gray')




