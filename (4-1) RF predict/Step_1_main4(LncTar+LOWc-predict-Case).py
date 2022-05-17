from dataset_construction import embedding2dataset
from dataset_construction import dump_the_model_RF
from dataset_construction import load_the_model_RF

from trans_test import calculate_metric
from trans_test import calculate_metric_notI
import os
import numpy as np
import pandas as pd
#Transfer from Project one to Project 2
#For project 1,Labels are dataA,dataB the features' name are featureA and feature B
#FeatureA and FeatureB are the descriptions of the features.
#For project 2,Labels are dataA, dataC, the features'name are featureA and featureC
#Model name means the model used
#Paras means the estimators=30,

list_auc=[]
list_aupr=[]
list_key=[]
for file_number in range(1,6):
  for file_number2 in range(1,6):
    project='LncTar+LC{}'.format(file_number)
    datasetA='.\\original_dataset\\constructed_lncTarD+LC_{}.csv'.format(file_number)
    project2='Case_study6(for LC+incTar)_{}'.format(file_number2)
    datasetB='.\\original_dataset\\constructed_CASE_STUDY_left_6_for LC+INCTar{}.csv'.format(file_number2)
 
    dataA='lnc'
    featureA='mix_kpca4096'
    dataB='gene'
    featureB='mix_kpca4096'
    dataC='gene'
    featureC='mix_kpca4096'
    model='RF'
    para='120'

    #output_name='.\\model\\{}-{}_{}-{}_{}-{}_{}-{}_{}-{}.m'.format(project,project2,dataA,featureA,dataB,featureB,dataC,featureC,model,para)
    output_name='.\\model\\{}-{}-{}.m'.format(project,model,para)
    file='.\\embedding_dataset\\{}-{}_{}-{}_{}.csv'.format(dataA,featureA,dataB,featureB,project)#AB for project1
    k=os.path.isfile(file)
    file2='.\\embedding_dataset\\{}-{}_{}-{}_{}.csv'.format(dataA,featureA,dataC,featureC,project2)#AC for project2
    k_f2=os.path.isfile(file2)
    model_file=output_name
    k2=os.path.isfile(model_file)
    print(k,k2)
    regenerate_flag=1
    if regenerate_flag==1:
     if k==False :
        print('generate the dataset from embedding files',dataA,dataB)
        X_train, y_train, X_test, y_test=embedding2dataset(fileA='.\\features\\lnc-palindormic_mix_kernalpolyPCA4096).csv',
                                                       labelA='label',
                                                       fileB='.\\features\\gene mix_kernelpolyPCA4096).csv',
                                                       labelB='label',
                                                       dataset=datasetA,
                                                       dataA=dataA,
                                                       dataB=dataB,
                                                       outputname=file,
                                                       partion=0.2,
                                                       randomseed=0)
     if k_f2==False:
         print('file not exist,generate the dataset from embedding files',dataA,'-',dataC)
         X_train, y_train, X_test, y_test=embedding2dataset(fileA='.\\features\\lnc-palindormic_mix_kernalpolyPCA4096).csv',
                                                       labelA='label',
                                                       fileB='.\\features\\\\gene mix_kernelpolyPCA4096).csv',
                                                       labelB='label',
                                                       dataset=datasetB,
                                                       dataA=dataA,
                                                       dataB=dataC,
                                                       outputname=file2,
                                                       partion=0.2,
                                                       randomseed=0)

    if k2!=True:
          print('begin to train and dump RF')
          dump_the_model_RF(dataset_name=file,
                           labelA=dataA,
                           labelB=dataB,
                           estimators=int(para),
                           output_name=output_name)
    print('model_existed',k2,'load',model_file)
    Y_pred,Ypb,Y_real=load_the_model_RF(original_dataset_name=file,
                                                #trans_dataset_name='.\\embedding_dataset\\{}_{}_{}.csv'.format(dataA,dataB,project),
    trans_dataset_name=file2,#embedding name
    trans_labelA='lnc',
    trans_labelB='gene',
    estimators=int(para),
    model_name=output_name)
    print(Ypb)
    #only need to add this to save the results.
    from dataset_construction import save_results
    save_results(dataset=file2,dataset_index=project2,model='LC+INC_dataset_case6{}'.format(file_number),labelA='lnc',labelB='gene',Y_real=Y_real,Y_predict=Ypb[:,1])
    calculate_metric(Y_real,Y_pred)
    #print('for columns 0')
    #print('using the first column')
    #calculate_metric_notI(Y_real,Ypb[:,0],1)
    print('using the second column')
    #print('for coulmns 1')c
    auc,aupr=calculate_metric_notI(Y_real,Ypb[:,1],1)
    list_auc.append(auc)
    list_aupr.append(aupr)
    list_key.append({'auc':auc,'aupr':aupr,'para':para,'file1':project,'file2':project2})
print('numbers of the records',len(list_key))
Aauc=round(np.mean(list_auc),4)
Aaupr=round(np.mean(list_aupr),4)
lk=pd.DataFrame(list_key)
lk.to_csv('.\\results\\Transfer_validation_report_of_Inctar+LC_trans_Case6_Mauc{}_Maupr{}(each Model).csv'.format(Aauc,Aaupr),index=None)
