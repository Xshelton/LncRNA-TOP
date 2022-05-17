import pandas as pd
import os  
allFileNum = 0
fileList=[]
def printPath(path):  
    global allFileNum 
    fileList = []
    files = os.listdir(path)
    for f in files:
        if(os.path.isfile(path + '/' + f)):  
            # 添加文件  
            fileList.append(f)
    print(fileList)
    return fileList
directory='.\gene ilearn'
fileList=printPath('.\gene ilearn')
def open_all_files(directory,label):
 flag=0
 for i in range(0,len(fileList)):
     doc=fileList[i]
     real_path = os.path.join(directory,doc)
     print(real_path)
     if flag==0:
        temp=pd.read_csv(real_path)
        label=temp['{}'.format(label)]
        print(temp.shape)
        temp=temp.drop(['label'],axis=1)
        print(temp.shape)
        flag=1
        print('initial',temp.shape)
     else:
        try:
         temp2=pd.read_csv(real_path)
         temp2=temp2.drop(['label'],axis=1)
         temp=pd.concat([temp,temp2],axis=1)
         flag+=1
         print('continue file',flag,temp.shape)
        except:
            print('error file',flag,'file corrupt',doc)
 return temp,label

from sklearn.decomposition import KernelPCA
dimension_list=[64,128,256,512,1024,2048,4096]
temp,label=open_all_files(directory,label='label')
project='gene-ilearn'
kernel='poly'#rbf

import numpy
for dimension in dimension_list:
    pca=KernelPCA(n_components=dimension, kernel=kernel, gamma=0.15, degree=2, coef0=0)
    temp1=numpy.nan_to_num(temp)#delete all of the Nan
    W=pca.fit_transform(temp1)
    print('the shape of W',W.shape)
    print(type(W))
    frame=pd.DataFrame(W)
    frame=pd.concat([frame,label],axis=1)
    frame.to_csv('{}_kernel{}PCA{}).csv'.format(project,kernel,dimension),index=None,encoding='utf8')
