# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 23:54:21 2022
@author: PAPLI
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd

import os

#path for train set image
path2=r"E:\sem6\ml\dataset_project\Train_set"

#path for csv file
path = r'E:\sem6\ml\dataset_project\Train_set\train.csv'

#read csv file
dataset = pd.read_csv(path)
print(dataset)
data=dataset.values[:7095,0:2]#7000 image for train




def limit(data):
#code for taking class 2 only
    i=0
    j=0
    k=0
    democlasses=[]
    demoimage=[]
    
    while(i!=245):
      if(data[j][1]==2):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    #code for taking class 1 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==1):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    #code for taking class 3 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==3):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
    
    
    #code for taking class 4 only
    i=0
    j=0
    while(i!=245):
      if(data[j][1]==4):
        democlasses.append(data[j][1])
        demoimage.append(data[j][0])
        i+=1
        j+=1
      else:
        j+=1
        
        
    return democlasses,demoimage


democlasses,demoimage=limit(data)


image=[]
classes=[]


#taking image from csv file
for i in range(246):   
  name=demoimage[i]
  join_path=os.path.join(path2,name)  
  img=cv2.imread(join_path)#read 5000 setof image
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(democlasses[i])#append all the class in classes
  
