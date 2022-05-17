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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout,MaxPooling2D
from tensorflow.keras import backend as K
import struct
import os
global bz
global epo
global dtz
import time
#from tfdeterminism import patch
#patch()
dtz=5#case number
bz=2
epo=50
print(keras.__version__, tf.__version__)
def load_dataset(dataset_name,labelA,labelB):
    print('load_the dataset')
    df=pd.read_csv('{}'.format(dataset_name))
    Y=df['label']
    df=df.drop(['{}'.format(labelA),'{}'.format(labelB),'label'],axis=1)
    X=df
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    Y1=onehot_encoded
    size=int(int(X.shape[1])**0.5)
    print('size of the embedding',size)
    size_of=int(size*size)
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
    return X_tuned,Y1,size
def dump_the_model_CNN(case_number,dataset_name,labelA,labelB,output_name,test_set_name,random_state):
    global bz
    global epo
    X_tuned,Y1,size=load_dataset(dataset_name,labelA,labelB)
    train_name=dataset_name[0:-4]
    partion=0.05
    rs=random_state
    #tf.random.set_seed(4487)
    tf.random.set_random_seed(rs)
    #tf.set_random_seed(rs)
    np.random.seed(rs)
    #random.seed(rs)
    #tf.global_variables_initializer().run()
    os.environ['PYTHONHASHSEED'] = str(rs)

    X_train,X_test,Y_train,Y_test=\
              model_selection.train_test_split(X_tuned, Y1, 
              train_size=1-partion, test_size=partion, random_state=rs,stratify=Y1)
    
    validsetI = (X_test, Y_test)
    K.clear_session()  # cleanup
    tf.reset_default_graph()
   
    nn = Sequential()
    nn.add(Conv2D(64, (1,4), strides=(1,1), activation='tanh',
    input_shape=(size,size,1),))
    nn.add(MaxPooling2D((1,2)))   
    nn.add(Conv2D(128, (2,2), strides=(1,1), activation='tanh',))
    nn.add(MaxPooling2D((1,2)))   
    #nn.add(Conv2D(256, (1,2), strides=(1,1), activation='tanh',))#S2
    #nn.add(MaxPooling2D((1,1)))   #S2
    nn.add(Flatten())
    #nn.add(Dense(units=512, activation='tanh', ))#S2
    nn.add(Dense(units=512, activation='tanh', ))
    nn.add(Dense(units=2, activation='sigmoid', ))
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',     # look at the validation loss
        min_delta=0.001,       # threshold to consider as no change
        patience=50,             # stop if 5 epochs with no change
        verbose=1, mode='auto'
    )
    callbacks_list = [earlystop]
    #callbacks_list=[]
    nn.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(lr=0.000002), 
               metrics=['accuracy'])
    # if you do the prediction , the don't have to consider the rs
    # And for the training part, it should use all of the dataset, thus the rs is not helpful

    history = nn.fit(X_tuned, Y1, epochs=epo, batch_size=bz, #difference is that here we use all of the dataset to train 
                     callbacks=callbacks_list, 
                     validation_data=validsetI, verbose=False,shuffle=False)
    
    
    model_save_path = ".\\model\\{}_{}_Structure_test_CS_epo{}_bz{}.h5".format(output_name,case_number,epo,bz)
    # 保存模型
    nn.save(model_save_path)
    print('model_save_successful')
    predY = nn.predict(X_test, verbose=False)
    auc=roc_auc_score(Y_test, predY)
    aupr=metrics.average_precision_score(Y_test, predY, average='macro', pos_label=1, sample_weight=None)
    print('test_auc_aupr',auc,aupr)
    list_case=[]
    for case in test_set_name:
        print(case)
        X_new_test,Y_new_test,size2=load_dataset(case,labelA,labelB)
        if size!=size2:
            print('size wrong')
        else:
         predY1 = nn.predict(X_new_test, verbose=False)

         auc1=roc_auc_score(Y_new_test, predY1)
         aupr1=metrics.average_precision_score(Y_new_test, predY1, average='macro', pos_label=1, sample_weight=None)
         print('validation_auc',auc,'validation_aupr',aupr,'new_test_auc_aupr',case,'auc',auc1,'aupr',aupr1)
         list_case.append({'train':dataset_name,'vali_auc':auc,'vali_aupr':aupr,'case':case,'auc':auc1,'aupr':aupr1,'rs':rs})
    return list_case
    
    #directly use another dataset, and see the results.
    
import pandas as pd
case_study_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_3.csv']
inc_tar_list=['lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap3.csv']
LC_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm1).csv','lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm2).csv','lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm3).csv']
test_set_list=[]
LC_INC_list=['lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC3.csv']
test_set_list=[]
def lncTar_predict_LC():
    list_case=[]
    #test_set
    #test_set_list=inc_tar_list+case_study_list

    # this project
    project='LC'
    test_set_list=LC_list
    #
    #project='Case_study'
    #test_set_list=case_study_list
    for case_number in range(3,4):
        for random_state in range(0,1):#for prediction, rs is not leveraged
          list_case=[]#每一次的尝试 都需要进行的操作是 清空list_case
          case='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap{}.csv'.format(case_number)
          list_case=list_case+dump_the_model_CNN(case_number,case,'lnc','gene','inc_tar_predict_LC',test_set_list,random_state=random_state)
          lc=pd.DataFrame(list_case)
          auc=round(lc['auc'].mean(),5)
          aupr=round(lc['aupr'].mean(),5)
          lc.to_csv('.\\results\\report_of_Inctar{}_(all)_predict_{}_rs{}_mAUC{}_mAUPR{}.csv'.format(case_number,project,random_state,auc,aupr),index=None)
def LC_predict_Inctar():
    list_case=[]
    #test_set
    #test_set_list=inc_tar_list+case_study_list

    # this project
    project='IncTar'
    test_set_list=inc_tar_list
    #
    #project='Case_study'
    #test_set_list=case_study_list
    for case_number in range(1,2):
        for random_state in range(0,1):#for prediction, rs is not leveraged
          list_case=[]#每一次的尝试 都需要进行的操作是 清空list_case
          case='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
          list_case=list_case+dump_the_model_CNN(case_number,case,'lnc','gene','LC_predict',test_set_list,random_state=random_state)
          lc=pd.DataFrame(list_case)
          auc=round(lc['auc'].mean(),5)
          aupr=round(lc['aupr'].mean(),5)
          lc.to_csv('.\\results\\report_of_LC{}_(all)_predict_{}_rs{}_mAUC{}_mAUPR{}.csv'.format(case_number,project,random_state,auc,aupr),index=None)

def LC_Inctar_predict():
    list_case=[]
    global bz
    global epo
    global dtz
    #test_set
    #test_set_list=inc_tar_list+case_study_list

    # this project
    project='LC_IncTar'
    test_set_list=case_study_list
    #
    #project='Case_study'
    #test_set_list=case_study_list

    for case_number in range(dtz,dtz+1):#by changing the 
        for random_state in range(0,1):#for prediction, rs is not leveraged
          list_case=[]#每一次的尝试 都需要进行的操作是 清空list_case
          case='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC{}.csv'.format(case_number)
          list_case=list_case+dump_the_model_CNN(case_number,case,'lnc','gene','LC_Inctar_predict',test_set_list,random_state=random_state)
          lc=pd.DataFrame(list_case)
          auc=round(lc['auc'].mean(),5)
          aupr=round(lc['aupr'].mean(),5)
          lc.to_csv('.\\results\\report_of_LC+IncTar_{}_(all)_predict_{}_rs{}_mAUC{}_mAUPR{}_epo{}_ba{}.csv'.format(case_number,project,random_state,auc,aupr,epo,bz),index=None)
def HC_predict():
    list_case=[]
    #test_set
    #test_set_list=inc_tar_list+case_study_list
    # this project
    project='HC'
    for case_number in range(dtz,dtz+1):
      test_set_list=['lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC{}.csv'.format(case_number)]
      for random_state in range(0,1):
           list_case=[]#每一次的尝试 都需要进行的操作是 清空list_case
           case='''lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC{}.csv'''.format(case_number)
           list_case=list_case+dump_the_model_CNN(case_number,case,'lnc','gene','HC_predict',test_set_list,random_state=random_state)
           lc=pd.DataFrame(list_case)
           auc=round(lc['auc'].mean(),5)
           aupr=round(lc['aupr'].mean(),5)
           lc.to_csv('.\\results\\report_of_HC_{}_(all)_predict_{}_rs{}_mAUC{}_mAUPR{}_epo{}_ba{}.csv'.format(case_number,project,random_state,auc,aupr,epo,bz),index=None)
start = time.time()
#different project

#lncTar_predict_LC()
#LC_predict_Inctar()
LC_Inctar_predict()
#HC_predict()
end = time.time()
print("The time used to execute this is given below")
print(end - start)
