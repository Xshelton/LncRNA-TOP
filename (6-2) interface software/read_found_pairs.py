import pandas as pd
import numpy
import numpy as np
global ref_list
global lnc_ensemble
global EN_ensemble
global gene_label
df=pd.read_csv('.//lnc_ensemble_index.csv')
ref=df['label']
ref_list=ref.values.tolist()
df2=pd.read_csv('.//gene_ensemble_index.csv')
gene_label=df2['label'].values.tolist()
df3=pd.read_csv('.//lncipedia_5_2_ensembl_92_genes.csv')
lnc_ensemble=df3['lncipediaGeneID'].values.tolist()
EN_ensemble=df3['ensemblGeneID'].values.tolist()

def return_file_name(index):
    global ref_list,lnc_ensemble,EN_ensemble
    try:
        #int(index)#directly using the index of 
        file_name='.//data//{}_{}_typeB_merge_3_merge_2.npy'.format(index,ref[index])
        #print(file_name)
    except:
        try:
            number_index=ref_list.index(index)#if index is a lnc ENSG
            file_name='.//data//{}_{}_typeB_merge_3_merge_2.npy'.format(number_index,ref[number_index])
            #print(file_name)
        except:
            try:
               print('Not index or ENSEMBLE ENSG000.x format,try find lnc ensemble')
               lnc_ensemble_index=lnc_ensemble.index(index)
               EN_ensemble_index=EN_ensemble[lnc_ensemble_index]
               number_index=ref_list.index(EN_ensemble_index)
               file_name='.//data//{}_{}_typeB_merge_3_merge_2.npy'.format(number_index,ref[number_index])
               #print(file_name)
            except:
               print('not match')
               file_name=None
    return file_name

def return_gene_file_name(index):
    global gene_label 
    try:
        file_name='.//data//{}_{}.npy'.format(index,gene_label[index])
    except:
        try:
             gene_index=gene_label.index(index)
             file_name='.//data//{}_{}.npy'.format(gene_index,index)
        except:
            print('gene ensemble not matched, please try again')
            file_name=None
    print(file_name)
    return file_name

        
def return_scores(A,B):
  global ref_list,gene_label
  f=return_file_name(A)
  if f!=None:
     data = numpy.load("{}".format(f))
     try:
         gene_index=gene_label.index('{}'.format(B))
         score=round(data[gene_index],5)
         key={'LncRNA':A,'gene':B,'scores':score}
     #print('score',score)
     except:
        key='No results Found'
    # key={}
     return key
    #\    print('gene not in the list')
def return_top_lnc(lnc,topk):
     global gene_label
     list_of_scores=[]
     f=return_file_name(lnc)
     if f!=None:
        data = numpy.load("{}".format(f))
        print(type(data),data.shape,'lnc_file_load_success')
        if len(data)==len(gene_label):
         for i in range(0,len(data)):
            key={'LncRNA':lnc,'gene':gene_label[i],'scores':data[i]}
            list_of_scores.append(key)
        list_of_scores = sorted(list_of_scores,key = lambda e:e['scores'],reverse = True)
     #print(list_of_scores[0:topk])
     return list_of_scores[0:topk]
        #gene_index=gene_label.index('{}'.format(B))

def return_top_gene(gene,topk):
     global gene_label,ref_list,lnc_ensemble,EN_ensemble
     list_of_scores=[]
     f=return_gene_file_name(gene)
     if f!=None:
        data = numpy.load("{}".format(f))
        print(type(data),data.shape,'gene_file_load_success')
        if len(data)==len(ref_list):
         for i in range(0,len(data)):
            lnc_index=EN_ensemble.index(ref_list[i])
            lnc=lnc_ensemble[lnc_index]
            key={'gene':gene,'LncRNA':ref_list[i],'LncRNA_ensemble':lnc,'scores':data[i]}
            list_of_scores.append(key)
        list_of_scores = sorted(list_of_scores,key = lambda e:e['scores'],reverse = True)
     print(list_of_scores[0:topk])
     return list_of_scores[0:topk]


    
def generate_gene_file():
    for i in range(2,len(gene_label)):#gene label  
        gene_index=i
        ensembl=gene_label[gene_index]
        file_name='{}_{}.npy'.format(gene_index,ensembl)
        if i%10==0:
          print(file_name)
        list_gene_score=[]
        for j in range(0,len(ref_list)):
             f=return_file_name(j)
             data = numpy.load("{}".format(f))
             temp_score=data[gene_index]
             #print(data.shape)
             list_gene_score.append(temp_score)
        lgs=np.array(list_gene_score)
        #print(lgs.shape)
        #np.save(file, arr
        np.save(file_name,lgs)
#generate_gene_file()

def return_key_words(keywords):
    global lnc_ensemble
    list_ref=[]
    for i,lnc in enumerate(lnc_ensemble):
        if keywords in lnc:
            list_ref.append(lnc)
    print(list_ref)
    return list_ref

#return_scores('DUBR','SIRT6')
#lnc='DUBR'
#return_top_lnc(lnc,20)
#gene='A1CF'
#return_gene_file_name(gene)
#return_top_gene(gene,topk=20)

#先看一下 lncRNA在不在里面
#再开一下 Gene在不在里面

#都在的话 就搜索一下这个lncRNA的文件 同时返回这个lncRNA预测结果中 gene的分数
