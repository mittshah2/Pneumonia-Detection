import numpy as np
import pandas as pd
import shutil,os
from tqdm import tqdm
import pydicom as dicom
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#change the paths according to your custom paths

df=pd.read_csv('C:\\Users\\mitts\\Desktop\\csv.csv')

pos=df[df['Target']==1]
pos['path']='C:\\Users\\mitts\\Desktop\\stage_2_train_images\\'+pos['patientId'].astype(str)+'.dcm'

un=pd.DataFrame({'path':pos['path'].unique(),'patientId': pos['patientId'].unique()})
unr=np.random.choice(un['path'],size=1200)




for i in tqdm(os.listdir('C:\\Users\\mitts\\Desktop\\train_images')):
    name=i[:-4]+'.txt'
    path=os.path.join('C:\\Users\\mitts\\Desktop\\train_images',name)
    fil = open(path, 'a')
    s = pos[pos['patientId'] == i[:-4]]
    for _,row in s.iterrows():
      st=str(0)+' '+str(row['x']/1024)+' '+str(row['y']/1024)+' '+str(row['width']/1024)+' '+str(row['height']/1024)+ '\n'  
      fil.write(st)
    fil.close()









