'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
import sys
print('train_MetaText   __name__ is ',__name__)
if __name__ == '__main__':
    REL_PATH = '../'
    sys.path.append(REL_PATH)
else:
    REL_PATH   = '../'
    
import sys, pickle, os,io, json
import numpy as np
from datetime                        import datetime
from classes.training_set            import TrainingSet
from sklearn.model_selection         import train_test_split
from Training                        import vectorize_TextData,smote_data
from Main.TrainingText               import MB_classifier, SGD_classifier, LR_classifier, GB_classifier
from Main.TrainingText               import SVC_classifier, LSVC_classifier
 
INPUT_PATH   = REL_PATH+'output/training/'
MODEL_PATH   = REL_PATH+'models/MetaText_models/'
'''
#--------------------------------------------------------------------------
#-- main() 
#--------------------------------------------------------------------------'''
def TrainMetaText(input_file):
    print(' --> Train MetaText started at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    pklfile = INPUT_PATH + input_file+'.pkl'  
    print('     Training set input file is :',pklfile)
    
    trainingSet = TrainingSet()
    trainingSet.load_from_Pkl(pklfile)
    ts_data = trainingSet.data
    ts_lbl = np.asanyarray(trainingSet.label , dtype = np.int)
    
    print('     Url length     :', len(trainingSet.url))       
    print('     Labels length  :', len(trainingSet.label))
    print('     Data length    :', len(trainingSet.data)) 
    print('     TagCount label :', trainingSet.label[:20])   
    print('     Multiclass Labels: ', ts_lbl.shape, np.bincount(ts_lbl))
    
#--------------------------------------------------------------------------
# Vectorize textual features using TfIdf vectorizer 
#-------------------------------------------------------------------------- 
    ts_data_sparse, ftr_names, vec_parms = vectorize_TextData(trainingSet,output=MODEL_PATH+'TfIdfVectorizer_Model.pkl')
    ts_data = ts_data_sparse.toarray()
    X_all = ts_data
    Y_all = ts_lbl 
 
#--------------------------------------------------------------------------
#  Perform oversampling of data using SMOTE    
#--------------------------------------------------------------------------
    X_all, Y_all = smote_data(ts_data, ts_lbl, 10, kind='regular')
    print(' Original Data shape     : ',ts_data.shape    , ' Original Label shape : ',ts_lbl.shape)
    print(' Oversampled Data shape  : ',X_all.shape , ' Oversampled Label shape : ',X_all.shape)
    
    
#--------------------------------------------------------------------------    
# Run Models 
#--------------------------------------------------------------------------
    # LR_classifier(X_all, Y_all, 'MetaText', train_ratio = 0.8, output_path = MODEL_PATH, TfIdf=False, NumFolds = 5)
    LSVC_classifier(X_all, Y_all, 'MetaText', train_ratio = 0.8, output_path = MODEL_PATH, TfIdf=False, NumFolds = 5)
    # GB_classifier(X_all, Y_all, 'MetaText', train_ratio = 0.8, output_path = MODEL_PATH, TfIdf=True, NumFolds = 5)
    # MB_classifier(X_all, Y_all, 'MetaText', train_ratio = 0.8, output_path = MODEL_PATH, TfIdf=True, NumFolds = 5)
    # MB_classifier(X_all, Y_all, 'MetaText', train_ratio = 0.8, output_path = MODEL_PATH, TfIdf=False, NumFolds = 5)
    
    print(' --> Train MetaText ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return

    
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------    
if __name__ == '__main__':
    
    meta_text_file = 'TRNFS_MetaText0507@1634'

    TrainMetaText(meta_text_file)
    exit(' MetaText Training Program completed successfully')
    