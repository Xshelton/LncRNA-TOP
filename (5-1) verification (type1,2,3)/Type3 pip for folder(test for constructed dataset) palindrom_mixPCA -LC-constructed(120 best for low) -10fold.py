# -*- coding: utf-8 -*-
#input the embedding file name of mirna/gene
#input the positive samples and negative samples dataset
#mode1:output the embedding file for the training/validating(one file)
#mode2:output the embedding file for the training80%/validating10%/testing%(two files)
#parameters seeds for np, mirna_file,gene_file,dataset_name
import pandas as pd
from numpy.random import seed
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score#从Sklearn指标中 引入准确率
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
global list_y_test
global list_name
global list_result
global list_color
#import keras
list_y_test=[]
list_name=[]
list_result=[]
list_color=[]
def clear_all_list():
    global list_y_test
    global list_name
    global list_result
    global list_color
    list_y_test=[]
    list_name=[]
    list_result=[]
    list_color=[] 
clear_all_list()
def roc_multiple(list_y_test,list_name,list_result,list_color):    
 print('开始载入图片')
 #plt.figure()
 #控制线的粗细
 plt.figure(figsize=(5,5))
 
 lw = 2
 print(len(list_name),len(list_result),len(list_result))
 for i in range(0,len(list_name)):
   y_test=list_y_test[i]
   fpr,tpr,threshold = roc_curve(y_test, list_result[i])
   precision,recall,thresholds = metrics.precision_recall_curve(y_test, list_result[i])
   
   roc_auc = auc(fpr,tpr)
   AUPR_=metrics.average_precision_score(y_test, list_result[i], average='macro', pos_label=1, sample_weight=None)
   plt.plot(fpr, tpr, color=list_color[i],lw=lw, label='{} (AUC = %0.2f)'.format(list_name[i]) % roc_auc)
   plt.plot(precision,recall, color=list_color[i],lw=lw, label='{} (AUPR = %0.2f)'.format(list_name[i]) % AUPR_)
 plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('Receiver operating characteristic curve')
 plt.legend(loc="lower right")
 plt.show()
def embedding2dataset(fileA,labelA,fileB,labelB,dataset,dataA,dataB,outputname,partion,randomseed):
    seed(randomseed)
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for mirna/gene')
    #try:
    df_dataset=pd.read_csv(dataset,encoding='gbk')
    D_A=df_dataset[dataA].values.tolist()
    D_B=df_dataset[dataB].values.tolist()
    D_label=df_dataset['label']
    print('step2,successfully load dataset for linc/gene/label')
    #except:
    #print('dataset load error:make sure the dataset with /{}/{}/label as columns header'.format(dataA,dataB))
    miss=0
    flag=0
    list_label=[]
    print('concating and generating the dataset,begin...')
    for i in range(0,len(df_dataset)):
        A_name=D_A[i]
        B_name=D_B[i]
        if A_name in dfA_label and B_name in dfB_label:
              A_index=dfA_label.index(A_name)
              B_index=dfB_label.index(B_name)
              featureA=dfA[A_index:A_index+1]
              featureA=featureA.reset_index(drop=True)
              featureB=dfB[B_index:B_index+1]
              featureB=featureB.reset_index(drop=True)
              if D_label[i]>0:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':1})
              else:
                         list_label.append({'{}'.format(dataA):A_name,'{}'.format(dataB):B_name,'label':0})
              if flag==0:#means the first time
                       temp0=pd.concat([featureA,featureB],axis=1,join='outer')
                       flag=1
              else:    #means the second time
                       temp1=pd.concat([featureA,featureB],axis=1,join='outer')
                       temp0=pd.concat([temp0,temp1],axis=0)
              if i%500==0 and i!=0:
                      if len(temp0)!=None:
                           print('Total generation',len(temp0),'/',len(D_A),'miss number',miss)
        else:
              miss+=1
              if miss!=0 and miss%500==0:
                      print('Miss_pairs_number_milestone',miss, '{}'.format(dataA),D_A[i],'{}'.format(dataB),D_B[i])
    print('concating and generating the dataset,finshed....','Total generation',len(temp0),'/',len(D_A))
    print('begin to cut dataset')
    fea_label=pd.DataFrame(list_label)
    X=temp0
    Y=fea_label
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=partion, random_state=randomseed)
    y_train=y_train.drop(['{}'.format(dataA)],axis=1)
    y_test=y_test.drop(['{}'.format(dataA)],axis=1)
    y_train=y_train.drop(['{}'.format(dataB)],axis=1)
    y_test=y_test.drop(['{}'.format(dataB)],axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    print('generate_file_for_future use')
    temp0=temp0.reset_index(drop=True)
    temp0=pd.concat([temp0,fea_label],axis=1)#这个feature 就包含了所有的抽取的正样本
    temp0.to_csv(outputname,index=None)
    return X_train, y_train, X_test, y_test,X,Y
           #Test_set=pd.concat([X_test,y_test],axis=1)
           #Test_set.to_csv('{}_test.csv'.format(outputname[0:-4]),index=None)#10% independent test set
def read_file(gg,filename):
   df=pd.read_csv(filename)
#gene	label	mirna
   label=df['label']
        #df=df.drop(['id'],axis=1)
   #df=df.drop(['0_mirna'],axis=1)
   if 'mirna' in df.columns.values:
    df=df.drop(['mirna'],axis=1)
    df=df.drop(['gene'],axis=1)
    df=df.drop(['label'],axis=1)
   if '0_mirna' in df.columns.values:
    df=df.drop(['0_mirna'],axis=1)
    df=df.drop(['1_gene'],axis=1)
    df=df.drop(['label'],axis=1)
   list_label2=[]
   for i in range(0,len(label)):
      if label[i]==1:
             list_label2.append(0)
      else:
             list_label2.append(1)
   list_2_pd=pd.DataFrame(list_label2)
   #label=pd.concat([label,list_2_pd],axis=1)
   Y = keras.utils.to_categorical(label)
   X_train, X_test, y_train, y_test,Y_train_onehot,y_test_onehot = train_test_split(df,label,Y,test_size=0.2, random_state=0)
   print('length of trainx',len(X_train))
   print('length of trainY',len(y_train))
   print('length of testX',len(X_test))
   print('length of testY',len(y_test))
   return X_train, X_test, y_train, y_test,Y_train_onehot,y_test_onehot,df,label

def append_list(name,y_test,y_predict,color):
    global list_y_test
    global list_name
    global list_result
    global list_color
    list_y_test.append(y_test)
    list_name.append(name)
    list_result.append(y_predict)
    list_color.append(color)
    
def Scaler(X_train,X_test):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
   # X_train=X_train.drop(['mirna'],axis=1)
   # X_test=X_test.drop(['mirna'],axis=1)
   # X_train=X_train.drop(['gene'],axis=1)
   # X_test=X_test.drop(['gene'],axis=1)
   # X_train=X_train.drop(['label'],axis=1)
    #X_test=X_test.drop(['label'],axis=1)
    X_train = scaler.fit_transform(X_train)
    #print(train_xgmmp.shape)
    X_test=scaler.transform(X_test)
    return X_train,X_test

def RF(X_train,y_train,X_test,y_test):#0.52
  from sklearn.ensemble import RandomForestClassifier
  clf5 = RandomForestClassifier(n_estimators=120)#
  #scaler = MinMaxScaler()
  scaler=StandardScaler()
  scaler.fit(X_train)
  X_train=scaler.transform(X_train)
  X_test=scaler.transform(X_test)
  clf5.fit(X_train,y_train)
  y_p=clf5.predict(X_test)
  acc = metrics.accuracy_score(y_test,y_p)
  print('RF',acc)
  y_pb=clf5.predict_proba(X_test)
  #print(y_pb[:,0])
  auc=roc_auc_score(y_test, y_pb[:,0])
  if auc<0.5:
      auc=roc_auc_score(y_test, y_pb[:,1])
 # auc=metrics.auc(y_test,y_pb[:,0])
  print('RF_AUC',auc)
  f1score=metrics.f1_score(y_test, y_p)
  print('RF_F1',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)
  print('RF MCC',MCC)
  macro_=metrics.average_precision_score(y_test, y_pb[:,0], average='macro', pos_label=1, sample_weight=None)
  if macro_<0.5:
      macro_=metrics.average_precision_score(y_test, y_pb[:,1], average='macro', pos_label=1, sample_weight=None)
  precision_e,recall_e,th=metrics.precision_recall_curve(y_test, y_pb[:,0])
  plt.plot(recall_e,precision_e, color = 'red')
  print("RF AUPR", macro_)
  print('RF:',metrics.classification_report(y_test,y_p))
  name='RF'
  #y_predict=y_score
  color='orange'
  append_list(name,y_test,y_p,color)
  return auc,macro_,f1score,MCC,metrics.classification_report(y_test,y_p)

def LR(X_train,y_train,X_test,y_test):#0.52
  import sklearn
  from numpy import logspace
  try:
    clf5 = sklearn.linear_model.LogisticRegressionCV(Cs=logspace(-2,2,3), cv=5, class_weight='unbalanced', solver='liblinear')
  except:
      from sklearn.linear_model import LogisticRegressionCV
      clf5=LogisticRegressionCV(Cs=logspace(-2,2,3), cv=5, class_weight='unbalanced', solver='liblinear')
  clf5.fit(X_train,y_train)
  y_p=clf5.predict(X_test)
  acc = metrics.accuracy_score(y_test,y_p)
  print('RF',acc)
  y_pb=clf5.predict_proba(X_test)
  #print(y_pb[:,0])
  auc=roc_auc_score(y_test, y_pb[:,0])
  if auc<0.5:
      auc=roc_auc_score(y_test, y_pb[:,1])
 # auc=metrics.auc(y_test,y_pb[:,0])
  print('fine_tuned_LR_AUC',auc)
  f1score=metrics.f1_score(y_test, y_p)
  print('fine_tunde_LR_F1',f1score)
  MCC=metrics.matthews_corrcoef(y_test, y_p)
  print('fine_tunde_LR MCC',MCC)
  macro_=metrics.average_precision_score(y_test, y_pb[:,0], average='macro', pos_label=1, sample_weight=None)
  if macro_<0.5:
      macro_=metrics.average_precision_score(y_test, y_pb[:,1], average='macro', pos_label=1, sample_weight=None)
  precision_e,recall_e,th=metrics.precision_recall_curve(y_test, y_pb[:,0])
  
  plt.plot(recall_e,precision_e, color = 'red')
  print("fine_tunde_LR AUPR", macro_)
  print('fine_tunde_LR:',metrics.classification_report(y_test,y_p))
  name='fine_tunde_LR'
  #y_predict=y_score
  color='orange'
  append_list(name,y_test,y_p,color)
  return auc,macro_,f1score,MCC,metrics.classification_report(y_test,y_p)

def RF10(X,Y,rs):
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import KFold
  clf5 = RandomForestClassifier(n_estimators=120)#
  #scaler = MinMaxScaler()
  cv=KFold(n_splits=10, shuffle=True, random_state=rs)
  from sklearn.model_selection import cross_val_score
  k=cross_val_score(clf5,X,Y,cv=cv,scoring='roc_auc')
  print('see, acutally, k contains 10 auc',k)
  auc=k.mean()
  k2=cross_val_score(clf5,X,Y,cv=cv,scoring='average_precision')
  aupr=k2.mean()
  return auc,aupr

def ada_5fold(X_train,y_train,X_test,y_test):
    from numpy import logspace
    from numpy import array
    from sklearn import model_selection
    from sklearn import ensemble
    paramgrid={'learning_rate':logspace(-5,5,10),'n_estimators':array([7,10,13,15,17,19])}
    #print(paramgrid)# setup the cross-validation object
    adacv=model_selection.GridSearchCV(ensemble.AdaBoostClassifier(random_state=0),paramgrid,cv=3,n_jobs=5,verbose=True)# run cross-validation (train for each split)
    adacv.fit(X_train,y_train);
    print("best params for adaboost:",adacv.best_params_)
    predY_ada= adacv.predict(X_test)
    print('shape',predY_ada.shape)
    auc=roc_auc_score(y_test, predY_ada)
 # auc=metrics.auc(y_test,y_pb[:,0])
    print('fine_tuned_LR_AUC',auc)
    f1score=metrics.f1_score(y_test, predY_ada)
    print('fine_tunde_LR_F1',f1score)
    MCC=metrics.matthews_corrcoef(y_test, predY_ada)
    print('fine_tunde_LR MCC',MCC)
    macro_=metrics.average_precision_score(y_test, predY_ada, average='macro', pos_label=1, sample_weight=None)
    print("fine_tunde_LR AUPR", macro_)
    acc_ada=metrics.accuracy_score(y_test,predY_ada)
    print("test accuracy for rf =",acc_ada)#0.6641
    print('Adaboost-5fold',metrics.classification_report(y_test,predY_ada))
    name='Adaboost-5fold'
    color='green'
    append_list(name,y_test,predY_ada,color)
    return auc,macro_,f1score,MCC,metrics.classification_report(y_test,predY_ada),adacv.best_params_
def count_0_1(y_train):
    y_tl=y_train.values.tolist()
    dict1 = {}
    for key in y_tl:
       dict1[key] = dict1.get(key, 0) + 1
    print(dict1)
def file_to_dataset(dataset_name,label1,label2,partion,rs):
   from numpy import random
   #random(0)
   randomseed=0
   seed(randomseed)
   df=pd.read_csv('{}'.format(dataset_name))
   print('successfully load embedding data file for mirna/gene')
   Y=df['label']
   df=df.drop(['{}'.format(label1),'{}'.format(label2),'label'],axis=1)
   X=df
   X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=partion,stratify=Y,random_state=rs)#fi
   print('the length of train_x',len(X_train))
   print('train_y',count_0_1(y_train))
   print('the length of test_x',len(X_test))
   print('test_y',count_0_1(y_test))
   #Train_set=pd.concat([X_train,y_train],axis=1)#contains 90% of the training set
   #Train_set.to_csv(outputname,index=None)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test=scaler.transform(X_test)
   print(type(X_train),'type of x_train')
   xtrainpd=pd.DataFrame(X_train)
   #xtrainpd.to_csv('temp_train.csv')
   return X_train,y_train,X_test,y_test,X,Y
#%%mirnafile,genefile,randomseed,dataset,outputname,partion):



#X_train,y_train,X_test,y_test=embedding2dataset('one_hot_plalindrome_mirna.csv','test_one_hot_plalindrome_gene_max15.csv',0,'Deepmirtar（ENSEMBLE）.csv','DeepmirTar_dataset_one_hot_for_both.csv',0.2) 
#X_train,y_train,X_test,y_test=embedding2dataset('plalindrome_mirna(PCA256).csv','plalindrome_gene(PCA512).csv',0,'Deepmirtar（ENSEMBLE）.csv','DeepmirTar_dataset_sg_pipline_test.csv',0.3) 
#X_train,y_train,X_test,y_test=embedding2dataset('plalindrome_mirna(PCA256).csv','plalindrome_gene(PCA512).csv',1,'SG128+miTarbase核心小数据集无人工(+,-).csv','SG128+miTarbase核心小数据集无人工(Pli_PCA_256_512).csv',0.2) 
#X_train,y_train,X_test,y_test=embedding2dataset('plalindrome_mirna(PCA256).csv','plalindrome_gene(PCA512).csv',0,'SG_核心小数据集_cos_only_FIVE.csv','SG_reliabe_dataset_cos+eu(distance).csv',0.9) 
#X_train,y_train,X_test,y_test=embedding2dataset('plalindrome_mirna(PCA256).csv','plalindrome_gene(PCA512).csv',0,'miRTarbase_8.0(human)_380639.csv','miRTarbase_8.0(human)_380639_PLI_PCA_256_512.csv',0.3) 
#X_train,y_train,X_test,y_test=embedding2dataset('SG128_mirna(R)_715.csv','SG128_gene_small_3068.csv',0,'SG_核心小数据集_cos_only_FIVE.csv','SG_reliabe_dataset_SG.csv',partion=0.1)
#X_train,y_train,X_test,y_test=embedding2dataset('mirna_doc2vec_34567_with_filter_hsa_2656.csv','gene_doc2vec_34567_mRNA.csv',0,'Deepmirtar（ENSEMBLE）.csv','Deepmirtar（ENSEMBLE）+SGnegative.csv',partion=0.4)
#filename='DeepmirTar_dataset_sg_pipline_test.csv'
#gg=1
#X_train, X_test, y_train, y_test,Y_train_onehot,y_test_onehot,df,label=read_file(gg,filename)
#X_train,X_test=Scaler(X_train,X_test)

#-----------------------
#fast load from file
#-----------------------

#
#X_train,y_train,X_test,y_test=file_to_dataset('DeepmirTar_dataset_one_hot_for_both.csv','mirna','gene',0.1)
 
#X_train,y_train,X_test,y_test=file_to_dataset('SG_核心小数据集_cos_only_FIVE.csv','mirna','gene',0.1)

#X_train,y_train,X_test,y_test=file_to_dataset('DeepmirTar_dataset_allmirna_gene_test_PCA6464.csv','mirna','gene',0.1)
#LR(X_train,y_train,X_test,y_test)

list_of_mp=[4096]
list_of_gp=[4096]
for index_mp in range(0,len(list_of_mp)):
 mp=list_of_mp[index_mp]
 gp=list_of_gp[index_mp]
 lncTarfile='.\lncRNA\lnrRNA_dataset\lncTarD\lncTar和incpedia之间的交集'
 low='''.\\lncRNA\\lnrRNA_dataset\\lncRNA2Target\\v3.0\\low throughput experiments\\lncRNA_target_from_low_throughput_experiments.csv和incpedia进行比较 对lncRNAid重构'''
 list_rf=[]
 list_auc=[]
 list_pr=[]
 frs=3
 for rs in range(0,10):
    for k in range(1,2):
            try:
             print('try to load dataset for training')
             X_train,y_train,X_test,y_test,X,Y=file_to_dataset('.\\Results\\temp\\constructed_LC(Remove)_rs{}_mkPCA{}mer_mkPCA{}.csv'.format(frs,mp,gp),'lnc','gene',0.1+0.1*k,rs)
             print('data load success')
            except:
             print('load_file_fail, try to generate file')#plalindrome_all_gene60_(kernel,PCA64).csv
                #fileA,labelA,fileB,labelB,dataset,dataA,dataB,outputname,partion,randomseed
             X_train,y_train,X_test,y_test,X,Y=embedding2dataset(fileA='.\\lncRNA\lncRNA_features\\mix PCA\\lnc-palindormic_mix_kernalpolyPCA{}).csv'.format(mp),#lnc-palindormic_kernelpolyPCA64).csv
                                                               labelA='label',
                                                               fileB='.\\gene\\gene_features\\mix kpca\\gene mix_kernelpolyPCA{}).csv'.format(gp),
                                                               labelB='label',
                                                               dataset='.\\Dataset construction\\manuelly curated negative samples\\constructed_LC(Remove)_rs{}.csv'.format(frs),
                                                               dataA='lnc',
                                                               dataB='gene',
                                                               outputname='.\\Results\\temp\\constructed_LC(Remove)_rs{}_mkPCA{}mer_mkPCA{}.csv'.format(frs,mp,gp),
                                                               partion=0.1+0.1*k,
                                                               randomseed=rs,
                                                               )
            
            #auc,macro_,f1score,MCC,report=RF(X_train,y_train,X_test,y_test)
            #lname='RF'
            #auc,macro_,f1score,MCC,report=LR(X_train,y_train,X_test,y_test)
            #lname='LR'
            #auc,macro_,f1score,MCC,report,best=ada_5fold(X_train,y_train,X_test,y_test)
            #lname='Ada'
            #list_auc.append(auc)
            #list_pr.append(macro_)
            #rf_key={'AUC':auc,'AUPR':macro_,'f1':f1score,'MCC':MCC,'portion_for_train':(1-0.1-0.1*k)}#'report':report,
            #list_rf.append(rf_key)
            
            auc,macro_=RF10(X,Y,rs)
            list_auc.append(auc)
            list_pr.append(macro_)
            lname='RF'
            #rf_key={'AUC':auc,'AUPR':macro_,'f1':f1score,'MCC':MCC,'portion_for_train':(1-0.1-0.1*k),'best_para_p(for_ada_only)':bp}#'report':report,
            rf_key2={'AUC':auc,'AUPR':macro_,'mode':'10-fold','model':lname}
            list_rf.append(rf_key2)
 lrf=pd.DataFrame(list_rf)
 print('temp save')
 from numpy import *
 m1=round(mean(list_auc),5)
 m2=round(mean(list_pr),5)
 print(mean(list_auc),mean(list_pr))
 lrf.to_csv('.\\Results\\Type3_{}_constructed_LC(Remove)_rs{}_(poly+poly 10folds)_Both_mix_kpca{}_{}portion_fix{}_{}_auc{}aupr_{}_epoch{}.csv'.format(lname,frs,mp,gp,(1-0.1-0.1*k),'',m1,m2,rs+1))
#ada_5fold(X_train,y_train,X_test,y_test)
#roc_multiple(list_y_test,list_name,list_result,list_color)
