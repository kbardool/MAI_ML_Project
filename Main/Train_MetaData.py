'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
import sys
print('train metadata1 __name__ is ',__name__)
if __name__ == '__main__':
    REL_PATH = '../'
    sys.path.append(REL_PATH)
else:
    REL_PATH   = '../'
    
import numpy as np
from datetime                          import datetime
from classes.training_set              import TrainingSet
from Training                          import smote_data, vectorize_Dict
from sklearn.model_selection           import train_test_split
from sklearn.feature_extraction        import DictVectorizer
from sklearn.preprocessing             import StandardScaler
from sklearn.externals                 import joblib
import Main.Train_MetaData2            as trn              
    
DEFAULT_PROCESS_COUNT = 9999
INPUT_PATH   = REL_PATH+'output/Training/'
OUTPUT_PATH  = REL_PATH+'output/'
SYSOUT_PATH  = REL_PATH+'output/'
MODEL_PATH   = REL_PATH+'models/MetaData_models/'
FILE_SFX = datetime.now().strftime("%m%d%Y@%H%M")
HTML_TEXT_FILE  = 'Tag_Count'


''' 
#--------------------------------------------------------------------------
#-- TrainMetaData
#--------------------------------------------------------------------------'''
def TrainMetaData(input_file = None): 
    print('\n\n -->  Train Meta Data started at: ', datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    pklfile = INPUT_PATH + input_file+'.pkl'  
    print(' Training set input file is :',pklfile)
    
    trainingSet = TrainingSet()
    trainingSet.load_from_Pkl(pklfile)

    print(' URLS length:    ', len(trainingSet.url))       
    print(' Labels length:  ', len(trainingSet.label))     
    print(' Data length:    ', len(trainingSet.data)) 
    print(' TagCount label : ', trainingSet.label[:20]) 
    ts_lbl  = np.asanyarray(trainingSet.label , dtype = np.int)


#--------------------------------------------------------------------------
# Vectorize features provided in form of dictionary  
#--------------------------------------------------------------------------
    vector = DictVectorizer()
    ts_sparse_data = vector.fit_transform(trainingSet.data)
    joblib.dump(vector, MODEL_PATH +'DictVectorizer.pkl')
    ts_data = ts_sparse_data.toarray()

    X_all = ts_data
    Y_all = ts_lbl
#--------------------------------------------------------------------------
#  Perform oversampling of data using SMOTE    
#--------------------------------------------------------------------------
    X_all, Y_all = smote_data(ts_data, ts_lbl, 10, kind='regular')
    print(' Original Data  shape : ', ts_data.shape)
    print(' Original Label shape : ', ts_lbl.shape)
    print(' X_all  Data  shape   : ', X_all.shape)
    print(' Y_alld Label shape   : ', Y_all.shape)
    
#--------------------------------------------------------------------------
# Standardize data features - Zero mean,Unit stddev
#--------------------------------------------------------------------------
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    joblib.dump(scaler, MODEL_PATH +'StandardScaler.pkl')
    
    # print(' Mean of all X data      : ', X_all.mean(axis=0))
    # print(' StdDev of all X data    : ', X_all.std(axis=0))                     
    # print(' X_all count:',len(X_all) ,' Y_all count:',len(Y_all)) 
    # print(' X_all shape:',X_all.shape,' Y_all shape:',Y_all.shape) 
 
#-------------------------------------------------------------------------- 
# Run Multi-class Training Algorithms 
#-------------------------------------------------------------------------- 
    # trn.LSVC_classifier(X_all, Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)  
    # trn.DT_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)   
    # trn.DT_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=True)   
    # trn.SGD_classifier(X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)   
    # trn.SVC_classifier(X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)    
    # trn.LR_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)   # ran 07-05 @ 2021
    # trn.GB_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)
    # trn.RF_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=False)
    # trn.RF_classifier( X_all,  Y_all, train_ratio = 0.8, output_path = MODEL_PATH, NumFolds=5,AdaBoost=True)      #with ADABOOST
 
 
    print(' --> Train Meta Data ended at:',datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return
    
    
#--------------------------------------------------------------------------
#  Driver 
#--------------------------------------------------------------------------
if __name__ == '__main__':
    
    input_file = 'TRNFS_MetaData0507@1634'
    TrainMetaData(input_file)
    exit(' TagCount Training Program completed successfully')

    
 

