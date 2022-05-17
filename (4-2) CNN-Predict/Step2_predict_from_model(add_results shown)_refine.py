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
from tensorflow.keras.models import load_model
LC_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm1).csv','lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm2).csv','lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm3).csv']
HC_list1=['lnc-mix_kpca4096_gene-mix_kpca4096_HC_refine(45331)_1.csv']
HC_list2=['lnc-mix_kpca4096_gene-mix_kpca4096_HC_refine(45331)_2.csv']
HC_list3=['lnc-mix_kpca4096_gene-mix_kpca4096_HC_refine(45331)_3.csv']
case_study_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_3.csv']
LC_special_case_study_list=['lnc-mix_kpca4096_gene-mix_kpca4096_LC_Case_study8_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LC_Case_study8_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LC_Case_study8_3.csv']
Lnc_special_case_study_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Case_study7_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study7_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study7_3.csv']
re_case_study_list=['lnc-mix_kpca4096_gene-mix_kpca4096_Case_study6(for LC+incTar)_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study6(for LC+incTar)_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study6(for LC+incTar)_3.csv']
inc_tar_list=['lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap3.csv']
#inctar 1,2,3 change the number here to change the original dataset+

def return_paras_inc_predict_case(case_number):
    ori_dataset='Inctar'
    #change predict projec#
    project='Case'
    test_set_name=Lnc_special_case_study_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\inc_tar_predict_LC_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_inc_predict_LC(case_number):
    ori_dataset='Inctar'
    #change predict projec#
    project='LC'
    test_set_name=LC_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\inc_tar_predict_LC_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name
    #change predict projec#
def return_paras_inc_predict_HC(case_number,HC_number):
    ori_dataset='Inctar'
    #change predict projec#
    #due to the memory limit, here the test_list shuold only be one
    project='HC'
    #test_set_name=HC_list1#LC_list#including the dataset of the test set
    test_set_name=['lnc-mix_kpca4096_gene-mix_kpca4096_High_throughput_constructed(rm{}).csv'.format(HC_number)]
    model_save_path = ".\\model\\inc_tar_predict_LC_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LC_predict_case(case_number):
    ori_dataset='LC'
    #change predict projec#
    project='Case'
    test_set_name=LC_special_case_study_list
    #case_study_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\LC_predict_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LC_predict_Inctar(case_number):
    ori_dataset='LC'
    #change predict projec#
    project='IncTar'
    test_set_name=inc_tar_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\LC_predict_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LC_predict_HC(case_number,HC_number):
    ori_dataset='LC'
    #change predict projec#
    #due to the memory limit, here the test_list shuold only be one
    project='HC'
    #test_set_name=HC_list1#LC_list#including the dataset of the test set
    test_set_name=['lnc-mix_kpca4096_gene-mix_kpca4096_High_throughput_constructed(rm{}).csv'.format(HC_number)]
    model_save_path = ".\\model\\LC_predict_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar_no_overlap{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LCINC_predict_case(case_number):
    ori_dataset='LC+inctar'
    #change predict projec#
    project='Case'
    test_set_name=case_study_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\LC_Inctar_predict_{}.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LCINC_predict_case(case_number):
    ori_dataset='LC+inctar'
    #change predict projec#
    project='Case'
    test_set_name=case_study_list
    #LC_list#including the dataset of the test set
    model_save_path = ".\\model\\LC_Inctar_predict_{}_Structure_test_CS.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_LncTar+LC{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LCINC_predict_HC(case_number,HC_number):
    ori_dataset='LC+inctar'
    #change predict projec#
    #due to the memory limit, here the test_list shuold only be one
    project='HC'
    #test_set_name=HC_list1#LC_list#including the dataset of the test set
    test_set_name=['lnc-mix_kpca4096_gene-mix_kpca4096_HC_refine(45331)_{}.csv'.format(HC_number)]
    model_save_path = ".\\model\\LC_Inctar_predict_{}_Structure_test_CS_epo50_bz2.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def return_paras_LCINC_predict_HCS2(case_number,HC_number):#MMO MMO.need big space
    ori_dataset='LC+inctar'
    #change predict projec#
    #due to the memory limit, here the test_list shuold only be one
    project='HCS2'
    #test_set_name=HC_list1#LC_list#including the dataset of the test set
    test_set_name=['lnc-mix_kpca4096_gene-mix_kpca4096_High_throughput_constructed(rm{}).csv'.format(HC_number)]
    model_save_path = ".\\model\\LC_Inctar_predict_{}_Structure_2.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name
def return_paras_LCINC_predict_case_S2(case_number):#
    ori_dataset='LC+inctar'
    #change predict projec#
    project='Case'
    test_set_name=case_study_list#LC_list#including the dataset of the test set
    model_save_path = ".\\model\\LC_Inctar_predict_{}_Structure_2.h5".format(case_number)
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_Low_throughput_constructed(rm{}).csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name
#change case_number to change the dataset of trained model..
def return_paras_LCLNC_predict_ALL(case_number):
    ori_dataset='LC+Inctar'
    project='All'
    model_save_path = ".\\model\\LC_Inctar_predict_{}_Structure_test_CS_epo50_bz2.h5".format(case_number)#LC_Inctar_predict_1_Structure_test_CS_epo50_bz2.h5
    test_set_name=['0']
    #using inctar predict LC. ori_dataset=,project=LC,test_set_name=LC_list'
    #using Inctar predict case. project=Case,test_set_name=case_study_list'
    dataset_name=''
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

def HC_predict_LC(case_number):
    ori_dataset='HC'
    #project==LC mean Using HC trained model for the LC
    project='LC'
    test_set_name=LC_list
    #---Project==IncTar====='
    #project='IncTar'
   # test_set_name=inc_tar_list
    #project='CASE_9'
    #test_set_name=['lnc-mix_kpca4096_gene-mix_kpca4096_Case_study9_for_HC_1.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study9_for_HC_2.csv','lnc-mix_kpca4096_gene-mix_kpca4096_Case_study9_for_HC_3.csv']
    model_save_path = ".\\model\\HC_predict_{}_Structure_test_CS_epo50_bz2.h5".format(case_number)
    print(model_save_path)
    dataset_name='lnc-mix_kpca4096_gene-mix_kpca4096_HC_refine(45331)_{}.csv'.format(case_number)
    return ori_dataset,project,test_set_name,model_save_path,dataset_name

 

case_number=5
HC_number=3
ori_dataset,project,test_set_name,model_save_path,dataset_name=return_paras_LCINC_predict_HC(case_number,HC_number)
#return_paras_LCLNC_predict_ALL(case_number)
print(test_set_name[0])
if 'HC' not in test_set_name[0] :
    print('HC not found',HC_number ,'reset into 0')
    HC_number=0

#---LC+Lnctar Predict HC dataset_case+HC_number---
#return_paras_LCINC_predict_HC(case_number,HC_number)

#---LC for train , HC for the test---
#return_paras_LC_predict_HC(case_number,HC_number)

#---LC for train , IncTar for the test---
#return_paras_LC_predict_Inctar(case_number)
    
#---LC for train , case for the test---
#return_paras_LC_predict_case(case_number)

#---IncTar predict High C---
#return_paras_inc_predict_HC(case_number,HC_number)

#--IncTar predict LC---   
#return_paras_inc_predict_LC(case_number)

#--IncTar predict Case--
#return_paras_inc_predict_case

random_state=0
#for the prediction,  random state are not applicable/
rs=random_state
labelA='lnc'
labelB='gene'
partion=0.05

    #tf.random.set_seed(4487)
tf.random.set_random_seed(rs)
    #tf.set_random_seed(rs)
np.random.seed(rs)
os.environ['PYTHONHASHSEED'] = str(rs)
def load_dataset(dataset_name,labelA,labelB):
    print('load_the dataset')
    df=pd.read_csv('{}'.format(dataset_name))
    Y=df.pop('label')
    labelA=df.pop('{}'.format(labelA))
    labelB=df.pop('{}'.format(labelB))
    
    results_form=pd.concat([labelA,labelB],axis=1)
    results_form['label']=Y
    
    # print('results form',results_form.shape)
    #print(df)
    #df=df.drop(['{}'.format(labelA),'{}'.format(labelB),'label'],axis=1)
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
    return X_tuned,Y1,size,results_form

    
if project!='All':
    print('initialize test mode')
    X_tuned,Y1,size,results_form_train=load_dataset(dataset_name,labelA,labelB)#This is the original dataset
    #This is the test set
    X_train,X_test,Y_train,Y_test=\
                  model_selection.train_test_split(X_tuned, Y1, 
                  train_size=1-partion, test_size=partion, random_state=rs,stratify=Y1)


    nn = load_model(model_save_path)#load the model
    predY = nn.predict(X_test, verbose=False)
    auc=roc_auc_score(Y_test, predY)
    aupr=metrics.average_precision_score(Y_test, predY, average='macro', pos_label=1, sample_weight=None)
    list_case=[]
    list_auc=[]
    list_aupr=[]
    for index,case in enumerate(test_set_name):#case are from the test_set_name
            print(case)
            X_new_test,Y_new_test,size2,results_form_2=load_dataset(case,labelA,labelB)
            
            if size!=size2:
                print('size wrong')
            else:
             predY1 = nn.predict(X_new_test, verbose=False)
             #print(results_form_2.shape)
             #print(predY1.shape)
             results_form_2['score_0']=predY1[:,0]
             results_form_2['score_1']=predY1[:,1]         
             auc1=roc_auc_score(Y_new_test, predY1)
             aupr1=metrics.average_precision_score(Y_new_test, predY1, average='macro', pos_label=1, sample_weight=None)
             list_auc.append(auc1)
             list_aupr.append(aupr1)
             print('validation_auc',auc,'validation_aupr',aupr,'new_test_auc_aupr',case,'auc',auc1,'aupr',aupr1)
             list_case.append({'train':dataset_name,'vali_auc':auc,'vali_aupr':aupr,'case':case,'auc':auc1,'aupr':aupr1,'rs':rs})
             print('writing to csv file')
             if HC_number==0:
                results_form_2.to_csv('.\\results\\individual results\\predictive_results{}_{}_{}_{}_auc_{}_aupr_{}.csv'.format(ori_dataset,case_number,project,index+1,round(auc1,5),round(aupr1,5)),index=None)
                results_form_2.to_csv('.\\results\\individual results\\CNN_predictive_results{}_{}_{}_{}.csv'.format(ori_dataset,case_number,project,index+1),index=None)
             else:
                 results_form_2.to_csv('.\\results\\individual results\\predictive_results{}_{}_{}_{}_auc_{}_aupr_{}.csv'.format(ori_dataset,case_number,project,HC_number,round(auc1,5),round(aupr1,5)),index=None)
                 results_form_2.to_csv('.\\results\\individual results\\CNN_predictive_results{}_{}_{}_{}.csv'.format(ori_dataset,case_number,project,HC_number),index=None)
    lc=pd.DataFrame(list_case)
    mauc=round(np.mean(list_auc),5)
    maupr=round(np.mean(list_aupr),5)
    if HC_number==0:
      lc.to_csv('.\\results\\test\\report_of_{}_{}_predict_{}_rs{}_mAUC{}_mAUPR{}.csv'.format(ori_dataset,case_number,project,random_state,mauc,maupr),index=None)
    else:
       lc.to_csv('.\\results\\test\\report_of_{}_{}_predict_{}_{}_rs{}_mAUC{}_mAUPR{}.csv'.format(ori_dataset,case_number,project,HC_number,random_state,mauc,maupr),index=None)  
elif project=='All':
      #load data from features first :
     print('initialize predict all mode')
     fileA='.\\features\\lnc-palindormic_mix_kernalpolyPCA4096).csv'
     labelA='label'
     fileB='.\\features\\\\gene mix_kernelpolyPCA4096).csv'
     labelB='label'
     dfA=pd.read_csv(fileA)
     dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
     dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
     dfB=pd.read_csv(fileB)
     dfB_label=dfB[labelB].values.tolist()
     dfB=dfB.drop([labelB],axis=1)
     print('step1,successfully load embedding data file for {}/{}'.format(labelA,labelB),'A:',len(dfA),'B:',len(dfB),'Total_predict',len(dfA)*len(dfB))    
     print(dfB.shape)#16127*4096
     nn = load_model(model_save_path)#load the model
     
     A_down=0
     A_up=len(dfA)
     #generate files one by one
     for A_index in range(A_down,A_up):
      if A_index%100==0:
        print(A_index,'out_of',len(dfA))
      list_tempB=[]
      #print('timer begin')
      #starttime = datetime.datetime.now()
      A_label=dfA_label[0:A_up]
      one_row_dfA=dfA[A_index:A_index+1]
      A_original_matrix=np.zeros((16127,4096))
      for i in range(0,len(A_original_matrix)):
        A_original_matrix[i]=one_row_dfA
      A_matrix=pd.DataFrame(A_original_matrix)
      AB=pd.concat([A_matrix,dfB],axis=1,join='outer')
      size=int(int(AB.shape[1])**0.5)
      #print('size of the embedding',size)
      size_of=int(size*size)
      try:
         npX=AB.values
      except:
         npX=AB
      if AB.shape[1]==size_of:#长度够 不要截断
        xpX=npX/255#否则需要截断/
      else:
        xpX=npX[:,:size_of]
      #print(xpX.shape)
      X_tuned=xpX.reshape(len(AB),size,size, 1)
      #print('the shape of A and B',X_tuned.shape)#16127 * 8192
      #model_save_path = ".\\model\\LC_Inctar_predict_1_Structure_test_CS.h5".format(case_number)
      
      y_pb=nn.predict(X_tuned,verbose=False)
      #score=y_pb[0][1]
      #print(y_pb)
      #print(y_pb[:,1])
      list_tempB=y_pb[:,1]
      import os
      from pathlib import Path
      my_file = Path(".//predict_results")
      if my_file.is_dir():
        #print('dir exist')
        pass
      else:
        print('not exist')
        os.makedirs(my_file)
        print('making a folder, finished')
      my_file2 = Path(".//predict_results//temp{}".format(case_number))
      if my_file2.is_dir():
        #print('dir exist')
         pass
      else:
        print('not exist')
        os.makedirs(my_file2)
        print('making a folder, finished')
      np.save('.//predict_results//temp{}//{}_{}_typeB'.format(case_number,A_index,A_label[A_index]),list_tempB)
      
