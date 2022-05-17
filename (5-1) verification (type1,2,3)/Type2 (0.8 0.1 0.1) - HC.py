import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from trans_test import calculate_metric_notI
from trans_test import calculate_metric
import numpy as np
#records
#1.constructed_dataset_low_throughput(ilearn-poly-ilearn-poly)_scaler_lnc_gene_kPCA{}mer_kPCA{}.csv'.format(dimensions,dimensions)
#2.constructed_dataset_low_throughput(rbfrbf)_scaler_lnc_gene_mkPCA64mer_mkPCA64.csv
#3.constructed_dataset_low_throughput_lnc_gene_mixPCA64mer_mixPCA64.csv
#4.constructed_dataset_LC_scaler_lnc_gene_mkPCA64mer_mkPCA64.csv
#5.去除没有的正样本 constructed_dataset_LC0(rm)_scaler_lnc_gene_mkPCA64mer_mkPCA64.csv
##rs=0
runtimes=10
list_key=[]
list_of_num=[4096]
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
for frs in range(2,3):#filename 1_2_3
  for num in range(0,len(list_of_num)):#
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    dimensions=list_of_num[num]
    list_test_auc=[]
    list_test_aupr=[]
    list_key=[]
    label1='lnc'
    label2='gene'
    dataset_name='.\\Model_predict\\ML\\embedding_dataset\\lnc-mix_kpca{}_gene-mix_kpca{}_High_throughput_constructed(rm{}).csv'.format(dimensions,dimensions,frs)#load the dataset
    df=pd.read_csv('{}'.format(dataset_name))
    print('successfully load embedding data file for lncRNA/gene','para=',dimensions)
    title='HC(rm{})_{}'.format(frs,dimensions)
    for rs in range(3,runtimes):
        #print(df.shape)
        Y=df['label']
        X=df.drop(['{}'.format(label1),'{}'.format(label2),'label'],axis=1)
        partion=0.2
        #X_train, X_test, y_train, y_test = \ train_test_split(X, Y,test_size=partion,stratify=Y,random_state=rs)#fi
        #X_test1=X_test.reset_index()
        #y_test1=y_test.reset_index()
        #print(X_test1.shape,y_test1.shape)
        #print(len(X_test),len(y_test))
        #X_vali,X_real_test,Y_vali,y_real_test=\ train_test_split(X_test1,y_test1,test_size=0.5,stratify=Y,random_state=rs)
        X_train,X_test,Y_train,Y_test=\
          model_selection.train_test_split(X, Y, 
          train_size=1-partion, test_size=partion, random_state=rs)
        scaler=StandardScaler()
        #scaler.fit_transform(X_train)
        #Type2: divide into 10 parts, 8 parts for train, 1 part fine-tune, 1 part test.
        X_train=scaler.fit_transform(X_train)
        
        X_vali,X_real_test,Y_vali,y_real_test = \
          model_selection.train_test_split(X_test, Y_test, 
          train_size=0.5, test_size=0.5, random_state=rs)
    
        X_vali=scaler.transform(X_vali)
        X_real_test=scaler.transform(X_real_test)   
        ##print(X_train.shape,valid_x.shape,real_test_x.shape)
        #print(1555+194+195)
        param_test1 = {'n_estimators':range(30,150,10),'min_samples_leaf':range(5,50,5),}#'max_depth':range(4,15,1)}#120,20,8
        gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,max_depth=8,max_features='sqrt' ,random_state=10), 
                               param_grid = param_test1, scoring='roc_auc',cv=5)
        gsearch1.fit(X_train,Y_train)
        #print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
        y_vali=gsearch1.predict(X_vali)
        #print('The results of the validation set')
        calculate_metric(Y_vali,y_vali)
        y_vali_prob=gsearch1.predict_proba(X_vali)
        Vauc,Vaupr=calculate_metric_notI(Y_vali,y_vali_prob[:,1],1)
        #print('The results of the test_set')
        Y_predict=gsearch1.predict(X_real_test)
        calculate_metric(y_real_test,Y_predict)
        y_predict_prob=gsearch1.predict_proba(X_real_test)
        auc,aupr=calculate_metric_notI(y_real_test,y_predict_prob[:,1],1)
        list_test_auc.append(auc)
        list_test_aupr.append(aupr)
        print(rs,'runtimes the end...')
        print(rs,'do the report')
        key={'validation_auc':Vauc,'validation_aupr':Vaupr,'test_auc':auc,'test_aupr':aupr,'random_state':rs,'best_para':gsearch1.best_params_}
        list_key.append(key)
        Aauc=round(np.mean(list_test_auc),4)
        Aaupr=round(np.mean(list_test_aupr),4)
        lk=pd.DataFrame(list_key)
        lk.to_csv('.\\Report_for_paper\\HCtype2\\TYPE2_{}_reports_of_0.8_0.1_0.1test_{}_auc{}_aupr{}.csv'.format(title,runtimes,Aauc,Aaupr),index=None)
   
