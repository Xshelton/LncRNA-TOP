import pandas as pd
import random

def load_positive_samples(filename,A,B):#读取正样本
    df=pd.read_csv(filename)
    nodeA=df['{}'.format(A)].values.tolist()
    nodeB=df['{}'.format(B)].values.tolist()
    posi_name=[]
    list_of_label=[]
    for i in range(0,len(df)):
        temp=nodeA[i]+'-'+nodeB[i]
        posi_name.append(temp)
        list_of_label.append(1)
    print((posi_name[0:10]))
    df['label']=list_of_label
    print(df)
    return df,posi_name
    

def add_positive_samples(filename,A,B,posi_name):
    print('Original',len(posi_name)) 
    df=pd.read_csv(filename)
    nodeA=df['{}'.format(A)].values.tolist()
    nodeB=df['{}'.format(B)].values.tolist()
    for i in range(0,len(df)):
        temp=nodeA[i]+'-'+nodeB[i]
        posi_name.append(temp)
        #list_of_label.append(1)
    print('New',len(posi_name))
    return posi_name


def load_ab_file(Afile,A,Bfile,B,outA,outB,posi_name,mount,random_seed):#load_file and generate enough negative samples
    dfA=pd.read_csv(Afile)
    dfB=pd.read_csv(Bfile)
    print('Afile_size',len(dfA))
    print('Bfile_size',len(dfB))
    nodeAs=dfA['{}'.format(A)].values.tolist()
    nodeBs=dfB['{}'.format(B)].values.tolist()
    #print(nodeAs,nodeBs)
    list_of_negative=[]
    random.seed(random_seed)    
    while len(list_of_negative)<=mount:
      if len(list_of_negative)%50==0:
          print(len(list_of_negative),'out of',mount)
      Ar=random.randint(0,len(dfA)-1)
      Br=random.randint(0,len(dfB)-1)
      #print(Ar,Br)
      nodeA=nodeAs[Ar]
      nodeB=nodeBs[Br]
      #print(nodeA,nodeB)
      check_if=nodeA+'-'+nodeB
      if check_if not in posi_name:
        list_of_negative.append({'{}'.format(outA):nodeA,'{}'.format(outB):nodeB,'label':0})#negative samples thus the 
    ln=pd.DataFrame(list_of_negative)
    print(ln)
    return ln

random_seed=5
df1,posi_name=load_positive_samples('.\original dataset\CASE_STUDY.csv','lnc','gene')
posi_name=add_positive_samples('.\original dataset\IncTarD_no_overlap.csv','lnc','gene',posi_name)
posi_name=add_positive_samples('.\original dataset\lncRNA_target_from_low_throughput_experiments_ENSG_both.csv','lnc','gene',posi_name)
posi_name=add_positive_samples('.\original dataset\CASE_STUDY.csv','lnc','gene',posi_name)
posi_name=add_positive_samples('.\original dataset\lncrna_target_high throughput experiments_v3_ENSG_both(rule out not existed).csv','lnc','gene',posi_name)


df2=load_ab_file('.\lnc\lnc-6mer_added_label.csv','label','.\gene\gene_3mer_added_label.csv','label',outA='lnc',outB='gene',posi_name=posi_name,mount=len(df1),random_seed=random_seed)
print(df2)
df3=pd.concat([df1,df2],axis=0)
print(df3.shape)
df3.to_csv('.\manuelly curated negative samples\constructed_case_study_rs{}.csv'.format(random_seed),index=None)
