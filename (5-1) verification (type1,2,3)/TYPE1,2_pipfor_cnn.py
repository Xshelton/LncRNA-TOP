
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout,MaxPooling2D
from tensorflow.keras import backend as K
import struct
import os
import time
print(keras.__version__, tf.__version__)
print('state of GPU',tf.test.is_gpu_available())

def plot_history(history, showacc=True): 
    fig, ax1 = plt.subplots()
    
    ax1.plot(history.history['loss'], 'r', label="training loss ({:.6f})".format(history.history['loss'][-1]))
    ax1.plot(history.history['val_loss'], 'r--', label="validation loss ({:.6f})".format(history.history['val_loss'][-1]))
    ax1.grid(True)
    ax1.set_xlabel('iteration')
    ax1.legend(loc="best", fontsize=9)    
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')

    if showacc and ('accuracy' in history.history):
        ax2 = ax1.twinx()

        ax2.plot(history.history['accuracy'], 'b', label="training acc ({:.4f})".format(history.history['accuracy'][-1]))
        ax2.plot(history.history['val_accuracy'], 'b--', label="validation acc ({:.4f})".format(history.history['val_accuracy'][-1]))

        ax2.legend(loc="best", fontsize=9)
        ax2.set_ylabel('acc', color='b')        
        ax2.tick_params('y', colors='b')
def LC_dataset(frs,dimensions):
    
#    df=pd.read_csv('.\\Results\\temp\\constructed_LC(Remove)_rs{}_mkPCA{}mer_mkPCA{}.csv'.format(frs,dimensions,dimensions))#Noisy dataset
    df=pd.read_csv('.\\Results\\temp\\constructed_LC(refine_493)_rs{}_mkPCA{}mer_mkPCA{}.csv'.format(frs,dimensions,dimensions))
    labels=df.pop('label')
    mirna=df.pop('lnc')
    gene=df.pop('gene')
    dts='LC'
    X=df
    Y=labels
    return X,Y,dts
def Inctar_dataset(frs,dimensions):
    #lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap2.csv
    #df=pd.read_csv('.\\Results\\temp\\lnc-mix_kpca{}_gene-mix_kpca{}_LncTar_no_overlap{}.csv'.format(dimensions,dimensions,frs))#Noisy Dataset
    df=pd.read_csv('.\\Results\\temp\\constructed_IncTar(refine_889)_rs_{}_{}.csv'.format(frs,dimensions))
    labels=df.pop('label')
    mirna=df.pop('lnc')
    gene=df.pop('gene')
    dts='LncTarD'
    X=df
    Y=labels
    return X,Y,dts
def LCInctarD_dataset(frs,dimensions):
    #lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap2.csv
    #df=pd.read_csv('.\\Results\\temp\\lnc-mix_kpca{}_gene-mix_kpca{}_LncTar_no_overlap{}.csv'.format(dimensions,dimensions,frs))#Noisy Dataset
    df=pd.read_csv('.\\Results\\temp\\constructed_ILC+IncTar(refine_1382)_rs_{}_4096.csv'.format(frs,dimensions))
    labels=df.pop('label')
    mirna=df.pop('lnc')
    gene=df.pop('gene')
    dts='LC_LncTarD'
    X=df
    Y=labels
    return X,Y,dts
#the number denotes the files code
#the second number denotes the dimensions

frs=1
#1,2,3 the dataset_file
#for each of the dataset, it has a
runtimes=10
#远程操作提示：需要继续跑3 然后从10到8到2 整个就可以把TYPE2给跑完
dimensions=4096
#2,4,6,8,10 this is to do the repeat, for each runtimes,it will set the rs(random state) as the runtime
#Thus in order to repeat, just change the runtimes, it would be okay.
#change dataset here/Inctar,Lowc
X,Y,dts=LCInctarD_dataset(frs,dimensions)
#LC_dataset(frs,dimensions)

model_type='TYPE2'
title='{}_{}_{}_4096_cnn'.format(model_type,dts,frs)




label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
Y1=onehot_encoded
print(X.shape[1])
print(Y1.shape)
size=int(int(X.shape[1])**0.5)
print('size of the embedding',size)
size_of=int(size*size)
print(size_of)
print(int(X.shape[1])**0.5)
try:
 npX=X.values
except:
 npX=X
if X.shape[1]==size_of:#长度够 不要截断
    xpX=npX/255#否则需要截断/
else:
    xpX=npX[:,:size_of]
print(xpX.shape)
X_tuned=xpX.reshape(len(X),size,size, 1)
list_test_auc=[]
list_test_aupr=[]
list_key=[]

for rs in range(runtimes-5,runtimes):
    #rs=0
    partion=0.2
    os.environ['PYTHONHASHSEED'] = str(rs)
    #for type1
    #0.8 for train(use itself to do the validation), 0.2 for test
    #repeat for 10 times, collect the test
    if model_type=='TYPE1':
        X_train,X_test,Y_train,Y_test=\
                  model_selection.train_test_split(X_tuned, Y1, 
                  train_size=1-partion, test_size=partion, random_state=rs,stratify=Y1)
        X_real_test=X_test
        y_real_test=Y_test
        X_vali=X_train
        Y_vali=Y_train
        validsetI =(X_vali,Y_vali)
        print('show the portion','type I')
        print('portion for train',len(X_train)/len(X_tuned),'portion for validation',len(X_train)/len(X_tuned),'portion for test',len(X_real_test)/len(X_tuned))
    #type 2/
    #0.8 for train,0.1 for validation(fine-tune),0.1 for the test
    #for each rs, the train, vali, test is different.
    if model_type=='TYPE2':
        start = time.time()
        print('show the portion','type II','0.8 train, 0.1validation, 0.1test')
        X_train,X_test,Y_train,Y_test=\
                  model_selection.train_test_split(X_tuned, Y1, 
                  train_size=1-partion, test_size=partion, random_state=rs,stratify=Y1)
                
        X_vali,X_real_test,Y_vali,y_real_test = \
                  model_selection.train_test_split(X_test, Y_test, 
                  train_size=0.5, test_size=0.5, random_state=rs,stratify=Y_test)
        validsetI = (X_vali, Y_vali)
        print('show the portion','type II')
        print('portion for train',len(X_train)/len(X_tuned),'portion for validation',len(X_vali)/len(X_tuned),'portion for test',len(X_real_test)/len(X_tuned))

    #print('for_train_shape',X_train.shape)
    #print('train_label',Y_train.shape)
    #print('validation_set_for_train_shape',X_vali.shape)
    #print('validation_set_for_test_shape',Y_vali.shape)
    #print('test_set_shape',X_real_test.shape)
    #print('test_y_shape',y_real_test)
    tf.random.set_random_seed(rs)   
    K.clear_session()  # cleanup
    tf.reset_default_graph()
    # initialize random seed
    #random.seed(4487); 
    #tf.random.set_seed(4487)
    
    
    nn = Sequential()
    nn.add(Conv2D(64, (1,4), strides=(1,1), activation='tanh',
                  input_shape=(size,size,1),))
    nn.add(MaxPooling2D((1,2)))   
    nn.add(Conv2D(128, (2,2), strides=(1,1), activation='tanh',))
    nn.add(MaxPooling2D((1,2)))   
    #nn.add(Conv2D(256, (1,2), strides=(1,1), activation='tanh',)) S2
    #nn.add(MaxPooling2D((1,1))) S2  
    nn.add(Flatten())
    nn.add(Dense(units=512, activation='tanh', ))
    #nn.add(Dense(units=512, activation='tanh', ))#S2
    nn.add(Dense(units=2, activation='sigmoid', ))
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',     # look at the validation loss
        min_delta=0.001,       # threshold to consider as no change
        patience=50,             # stop if 5 epochs with no change
        verbose=1, mode='auto'
    )
    callbacks_list = [earlystop]
    epo=50
    bz=2
    nn.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.RMSprop(lr=0.000002), 
              metrics=['accuracy'])
    history = nn.fit(X_train,Y_train, epochs=epo, batch_size=bz, 
                     callbacks=callbacks_list, 
                     validation_data=validsetI, verbose=False)
    #save the model
    # For type 1,2 and 3, no need to do the dump the model
    save_mode=0
    if save_mode==1:
      model_save_path = ".\\Reports_for_paper_Cnn\\model\\{}_{}_rs{}.h5".format(model_type,title,rs)
      nn.save(model_save_path)
    
    #plot_history(history)
    predY = nn.predict(X_vali, verbose=False)
    auc=roc_auc_score(Y_vali, predY)
    aupr=metrics.average_precision_score(Y_vali, predY, average='macro', pos_label=1, sample_weight=None)
     # auc=metrics.auc(y_test,y_pb[:,0])
    
    #print('fine_tuned_CNN_AUC_vali',auc)
    #print('------------------------------')
    #print('fine_tuned_CNN_AUPR(Validation)',aupr)
    predY2=nn.predict(X_real_test, verbose=False)
    #print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    auc2=roc_auc_score(y_real_test,predY2)
    aupr2=metrics.average_precision_score(y_real_test,predY2, average='macro', pos_label=1, sample_weight=None)
    #print('fine_tuned_CNN_AUC(X_Test)',auc2)
    #print('------------------------------')
    #print('fine_tuned_CNN_AUPR(X_Test)',aupr2)
    end = time.time()
    print("The time used to execute this is given below")
    print(end - start)
    dur=end - start
    list_test_auc.append(auc2)
    list_test_aupr.append(aupr2)
    key={'validation_auc':auc,'validation_aupr':aupr,'test_auc':auc2,'test_aupr':aupr2,'random_state':rs,'during':dur}
    list_key.append(key)
    
Aauc=round(np.mean(list_test_auc),4)
Aaupr=round(np.mean(list_test_aupr),4)
lk=pd.DataFrame(list_key)

if model_type=='TYPE1':
   lk.to_csv('.\\Reports_for_paper_Cnn\\{}_rs{}_auc{}_aupr{}_{}_{}.csv'.format(title,runtimes,Aauc,Aaupr,epo,bz),index=None)
if model_type=='TYPE2':
   lk.to_csv('.\\Reports_for_paper_Cnn\\{}_rs{}_auc{}_aupr{}_{}_{}.csv'.format(title,runtimes,Aauc,Aaupr,epo,bz),index=None)
