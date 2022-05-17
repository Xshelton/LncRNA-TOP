from numpy.random import seed
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy
import datetime
import numpy as np
#from public_function import check_create
#joblibb.dump(model,loca)
#joblib.load(x
import pandas as pd
def check_create(folder_name):
  my_file = Path(".\{}".format(folder_name))
  if my_file.is_dir():
    print('dir exist')
  else:
    print('not exist')
    os.makedirs(my_file)
    print('making a folder, finished')
    
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
    print('step2,successfully load dataset for lnc/gene/label',len(D_A),len(D_B))
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
    return X_train, y_train, X_test, y_test

def file_to_dataset(dataset_name,labelA,labelB,partion,randomseed2):
   randomseed=0
   seed(randomseed)
   df=pd.read_csv('{}'.format(dataset_name))
   print('successfully load embedding dataset file from{}'.format(dataset_name))
   Y=df['label']
   df=df.drop(['{}'.format(label1),'{}'.format(label2),'label'],axis=1)
   X=df
   print('divide the file into X_train,X_test')      
   X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=partion,stratify=Y,random_state=randomseed2)#fi
   #print('the length of train_x',len(X_train))
   #print('train_y',count_0_1(y_train))
   #print('the length of test_x',len(X_test))
   #print('test_y',count_0_1(y_test))
   #Train_set=pd.concat([X_train,y_train],axis=1)#contains 90% of the training set
   #Train_set.to_csv(outputname,index=None)
   #scaler = StandardScaler()
   #X_train = scaler.fit_transform(X_train)
   #X_test=scaler.transform(X_test)
   #print(type(X_train),'type of x_train')
   #xtrainpd=pd.DataFrame(X_train)
   #xtrainpd.to_csv('temp_train.csv')
   return X_train,y_train,X_test,y_test
def dump_the_model_RF(dataset_name,labelA,labelB,estimators,output_name):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    print('read_dataset_file')
    df=pd.read_csv('{}'.format(dataset_name))
    print(df.columns)
    Y=df['label']
    df=df.drop(['{}'.format(labelA),'{}'.format(labelB),'label'],axis=1)
    X=df
    clf_final = RandomForestClassifier(n_estimators=estimators)
    clf_final.fit(X,Y)
    print('model_training success')
    joblib.dump(clf_final,output_name)
    print('model saving success')
    
def load_the_model_RF(original_dataset_name,trans_dataset_name,trans_labelA,trans_labelB,estimators,model_name):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    print('read_dataset_file')
    df_original=pd.read_csv('{}'.format(original_dataset_name))
    df=pd.read_csv('{}'.format(trans_dataset_name))
    print(df.columns)
    if df_original.iloc[0].shape!=df.iloc[0].shape:
      print(df_original.iloc[0].shape,df[0].iloc.shape,'shape not the same,abortion')
    else:
      print(df_original.iloc[0].shape,df.iloc[0].shape,'shape the same, transfer begin')
      Y=df['label']
      df=df.drop(['{}'.format(trans_labelA),'{}'.format(trans_labelB),'label'],axis=1)
      X=df
      clf_final = joblib.load('{}'.format(model_name)) #调用
      print('load file sccuess')
      y_predict=clf_final.predict(X)
      print('generate_Y_predict,successfully')
      y_pb=clf_final.predict_proba(X)
    return y_predict,y_pb,Y#返回真实情况 return predic and real label

def return_out(frs):
          file_number=frs
          file_number2=frs
          #LncTar+LC(refine1382)_rm2-HC(refine45392)(rm3)_lnc-mix_kpca64_gene-mix_kpca64_gene-mix_kpca64_RF-120.m
          #LncTar+LC1-Case_study6(for LC+incTar)_1_lnc-mix_kpca4096_gene-mix_kpca4096_gene-mix_kpca4096_RF-120.m
          project='LncTar+LC(refine1382)_rm{}'.format(file_number)
          datasetA='.\\original_dataset\\constructed_lncTarD+LC_{}.csv'.format(file_number)
          project2='High_throughput_constructed(rm{})'.format(file_number2)
          datasetB='.\\original_dataset\\constructed_HC_rm{}.csv'.format(file_number2)
       
          dataA='lnc'
          featureA='mix_kpca4096'
          dataB='gene'
          featureB='mix_kpca4096'
          dataC='gene'
          featureC='mix_kpca4096'
          model='RF'
          para='120'

          output_name='.\\model\\{}-{}_{}-{}_{}-{}_{}-{}_{}-{}.m'.format(project,project2,dataA,featureA,dataB,featureB,dataC,featureC,model,para)
          return output_name
        
def generate_to_predict(fileA,labelA,fileB,labelB,model_name):#
    import joblib
    clf_final = joblib.load('{}'.format(model_name)) #调用
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for {}/{}'.format(labelA,labelB),'A:',len(dfA),'B:',len(dfB),'Total_predict',len(dfA)*len(dfB))
    A_index=0
    B_index=0
    A_down=0
    A_up=len(dfA)
    B_up=len(dfB)
    list_score=[]
    print('step2','begin to predict')
    for A_index in range(A_down,A_up):
      starttime = datetime.datetime.now()
      print('begin timer')
      if A_index%500==0:
        print(A_index,'out_of',A_up)
      A_label=dfA_label[0:A_up]
      list_temp_B=[]
      for B_index in range(0,B_up):
        B_label=dfB_label[0:B_up]
        if B_index%500==0:
          print('A_index',A_index,A_label[A_index],'B_index:',B_index,B_label[B_index],)
        
        featureA=dfA[A_index:A_index+1]
        featureA=featureA.reset_index(drop=True)
        featureB=dfB[B_index:B_index+1]
        featureB=featureB.reset_index(drop=True)
        temp0=pd.concat([featureA,featureB],axis=1,join='outer')
        y_pb=clf_final.predict_proba(temp0)
        #print(type(y_pb))
        score=y_pb[0][1]
        #print('predict_results',y_pb,score)
        list_temp_B.append(score)
        
      print('B_temp_save')
      lbnp=numpy.array(list_temp_B)
      numpy.save('.//predict_results//temp//{}_{}'.format(A_index,A_label[A_index],list_temp_B),lbnp)
      list_score.append(list_temp_B)
      endtime = datetime.datetime.now()
      print('end_timer')
      print('time for predicting one lnc',(endtime-starttime))
      #10:24.622682 10min for one lnc too slow.
    #print(list_score)
    lsnp=numpy.array(list_score)
    #print(lsnp.shape)
    #print(lsnp)
    numpy.save('.//predict_results//inc+LC_predict_{}_{}_total_pairs_{}'.format(A_up,B_up,A_up*B_up),lsnp)
    lspd=pd.DataFrame(lsnp)
    print(lspd.shape)
    print(len(A_label),len(B_label))
    lspd.columns=B_label
    lspd.index=A_label
    lspd.to_csv('.//predict_results//predict_results_{}_{}_total_{}.csv'.format(A_up,B_up,A_up*B_up))

def generate_to_predict_type2(fileA,labelA,fileB,labelB,model_name):
    import joblib
    clf_final = joblib.load('{}'.format(model_name)) #调用
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for {}/{}'.format(labelA,labelB),'A:',len(dfA),'B:',len(dfB),'Total_predict',len(dfA)*len(dfB))    
    print(dfB.shape)#16127*4096
    A_down=0
    A_up=len(dfA)
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
      #print(AB.shape)#16127 * 8192
      y_pb=clf_final.predict_proba(AB)
      #score=y_pb[0][1]
      #print(y_pb)
      #print(y_pb[:,1])
      list_tempB=y_pb[:,1]
      numpy.save('.//predict_results//temp3-3//{}_{}_typeB'.format(A_index,A_label[A_index]),list_tempB)
      #endtime = datetime.datetime.now()
      #print('end_timer')
      #print('time for predicting one lnc',(endtime-starttime))
def generate_to_predict_type3(fileA,labelA,fileB,labelB,model_name,out_dir):
    import joblib
    clf_final = joblib.load('{}'.format(model_name)) #调用
    print('step0,succesfully reload the model')
    dfA=pd.read_csv(fileA)
    dfA_label=dfA[labelA].values.tolist()#fileA的标签 labels of file A
    dfA=dfA.drop([labelA],axis=1)#fileA的特征 features of fileA
    dfB=pd.read_csv(fileB)
    dfB_label=dfB[labelB].values.tolist()
    dfB=dfB.drop([labelB],axis=1)
    print('step1,successfully load embedding data file for {}/{}'.format(labelA,labelB),'A:',len(dfA),'B:',len(dfB),'Total_predict',len(dfA)*len(dfB))    
    print(dfB.shape)#16127*4096
    A_down=0
    A_up=len(dfA)
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
      #print(AB.shape)#16127 * 8192
      y_pb=clf_final.predict_proba(AB)
      #score=y_pb[0][1]
      #print(y_pb)
      #print(y_pb[:,1])
      list_tempB=y_pb[:,1]
      numpy.save('{}//{}_{}_typeB'.format(out_dir,A_index,A_label[A_index]),list_tempB)
      

def return_label_pair(dataset,labelA,labelB):
      print('load_dataset',dataset)
      df=pd.read_csv(dataset)
      print('dataset_load suc')
      list_labelA=df.pop(labelA)
      list_labelB=df.pop(labelB)
      df_pair=pd.concat([list_labelA,list_labelB],axis=1)
      print(df_pair.shape)
      #print(df_pair)
      return df_pair
def save_results(dataset,dataset_index,model,labelA,labelB,Y_real,Y_predict):
  df_pair=return_label_pair(dataset,labelA,labelB)
  df_pair['label']=Y_real
  df_pair['score']=Y_predict
  df_pair.to_csv('.\\individual results\\{}_predict_{}_scores.csv'.format(model,dataset_index),index=None)
#return_label_pair(dataset='.\\embedding_dataset\\lnc-mix_kpca4096_gene-mix_kpca4096_Case_study_1.csv',labelA='lnc',labelB='gene')

#generate_to_predict_type2(fileA='.\\features\\lnc-palindormic_mix_kernalpolyPCA4096).csv',
#                                                       labelA='label',
#                                                       fileB='.\\features\\\\gene mix_kernelpolyPCA4096).csv',
#                                                       labelB='label',
#                                                      model_name=return_out(3))
#check_create('model')


#embedding2dataset(fileA=,labelA=,fileB=,geneB=,dataset=,outputname=,partion=,randomseed=)
