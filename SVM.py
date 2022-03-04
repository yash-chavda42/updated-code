import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
import cv2
import os
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#path for train set image
path2=r"E:\sem6\ml\dataset_project\Train_set"

#path for csv file
path = r'E:\sem6\ml\dataset_project\Train_set\train.csv'

#read csv file
dataset = pd.read_csv(path)
print(dataset)
data=dataset.values[:7095,0:2]#7000 image for train


image=[]
classes=[]

#taking image from csv file
for i in range(100):   
  name=data[i][0]
  join_path=os.path.join(path2,name)  
  img=cv2.imread(join_path)#read 5000 setof image
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(data[i][1])#append all the class in classes
  


#converitng into numpy
image=np.array(image)
classes=np.array(classes)




#split train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image, classes, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = x_test.shape
x_test = x_test.reshape((nsamples,nx*ny))

print(x_test.shape)
print(x_train.shape)


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)

#for train
pred_class_train = clf.predict(x_train)
print(y_train)
print(pred_class_train)

print (classification_report(y_train, pred_class_train))


#test
pred_class_test = clf.predict(x_test)
pred_class_test 

print (classification_report(y_test, pred_class_test))

