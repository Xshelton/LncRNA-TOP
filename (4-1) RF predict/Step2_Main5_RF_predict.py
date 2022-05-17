from dataset_construction import return_out
from dataset_construction import generate_to_predict_type3
frs=3
#LncTar+LC1-RF-120.m
model_name='.\\model\\LncTar+LC{}-RF-120.m'.format(frs)
generate_to_predict_type3(fileA='.\\features\\lnc-palindormic_mix_kernalpolyPCA4096).csv',
                                                       labelA='label',
                                                       fileB='.\\features\\\\gene mix_kernelpolyPCA4096).csv',
                                                       labelB='label',
                                                       model_name=model_name,
                                                       out_dir='.//predict_results//refine_temp{}-{}'.format(frs,frs)  )
