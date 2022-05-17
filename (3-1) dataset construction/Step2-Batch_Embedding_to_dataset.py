import pandas as pd
from numpy.random import seed
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
              if i%2500==0 and i!=0:
                      if len(temp0)!=None:
                           print('Total generation',len(temp0),'/',len(D_A),'miss number',miss)
        else:
              miss+=1
              if miss!=0 and miss%2500==0:
                      print('Miss_pairs_number_milestone',miss, '{}'.format(dataA),D_A[i],'{}'.format(dataB),D_B[i])
    print('concating and generating the dataset,finshed....','Total generation',len(temp0),'/',len(D_A))
    print('begin to cut dataset')
    fea_label=pd.DataFrame(list_label)
    X=temp0
    Y=fea_label
    print('generate_file_for_future use')
    temp0=temp0.reset_index(drop=True)
    temp0=pd.concat([temp0,fea_label],axis=1)#这个feature 就包含了所有的抽取的正样本
    temp0.to_csv(outputname,index=None)
mp=4096
gp=4096
for frs in range(4,6):
 rs=frs
 embedding2dataset(fileA='.\\lnc\lnc-palindormic_mix_kernalpolyPCA{}).csv'.format(mp),#lnc-palindormic_kernelpolyPCA64).csv
                                                               labelA='label',
                                                               fileB='.\\gene\\gene mix_kernelpolyPCA{}).csv'.format(gp),
                                                               labelB='label',
                                                               dataset='.\\manuelly curated negative samples\\constructed_LC+IncTar(refine_1382)_rs{}.csv'.format(frs),
                                                               dataA='lnc',
                                                               dataB='gene',
                                                               outputname='.\\Embedding_dataset\\constructed_ILC+IncTar(refine_1382)_rs_{}_{}.csv'.format(frs,mp),
                                                               partion=0.1,
                                                               randomseed=rs,
                                                               )
