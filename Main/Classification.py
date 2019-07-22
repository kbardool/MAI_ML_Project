'''
Created on Apr 12, 2017

@author: kevin.bardool
'''

# print('train Classifier __name__ is ',__name__)  
import sys
if __name__ == '__main__':
    REL_PATH = '..//'
    OUTPUT_PATH       = REL_PATH+'output/Classify/'
    INPUT_PATH        = REL_PATH+'output/Test/'
    sys.path.append(REL_PATH)
else:
    REL_PATH    = './/'
    INPUT_PATH  = ''
    OUTPUT_PATH = ''
    
import  os , io, time, csv
import  argparse
from datetime               import datetime
from common.LABELS          import LABEL_NAMES
from nltk.metrics.aline     import np
from sklearn.externals      import joblib
from classes.training_set   import TrainingSet
from Main.Training          import vectorize_Dict
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble       import  VotingClassifier
# from Main.Train_MetaData2 import *
# from classes.training_set            import TrainingSet


 
MODEL_PATH        = {'MetaData' : REL_PATH+'models//MetaData_Models//',
                     'HtmlText' : REL_PATH+'models//HtmlText_Models//',
                     'MetaText' : REL_PATH+'models//MetaText_Models//'}
                     
def classify_HtmlText(HtmlText, Models, output_file = None):
    """
    --------------------------------------------------------------------------------
    Process HtmlText FeatureData
    --------------------------------------------------------------------------------
    """  
    X1 = HtmlText    
    print('     ------------------------------------------------------------------------------')
    print('      Classify HtmlText Feature set')
    print('     ------------------------------------------------------------------------------')

    print('      URLS length:    ', len(X1.url))      
    print('      Data length:    ', len(X1.data)) 
    Vectr1 = joblib.load(MODEL_PATH['HtmlText']+'LR_TfIdfVectorizer.pkl')
    ts_sparse_data = Vectr1.transform(X1.data)
    X1_data = ts_sparse_data.toarray()   
    
    y1_pred       = Models[0].predict(X1_data)
    y1_pred_name  = [LABEL_NAMES[i] for i in y1_pred]
    
    if output_file is not None:
        output = open(output_file, 'w' ,newline ='', encoding='utf-8')
        output_csv = csv.writer(output ,quoting=csv.QUOTE_MINIMAL)
        for url, pred in zip(X1.url, y1_pred):
            output_csv.writerow(url.split('.')+ [LABEL_NAMES[pred]])
        output.close()
        
    print_results(X1, y1_pred, 'Html Text')
    return
                     
def classify_MetaText(MetaText, Models, output_file = None):
    """
    --------------------------------------------------------------------------------
    Process MetaText FeatureData
    --------------------------------------------------------------------------------
    """     
    X2 = MetaText
    print('     ------------------------------------------------------------------------------')
    print('      Classify MetaText Feature set')
    print('     ------------------------------------------------------------------------------')    
    print('      URLS length:    ', len(X2.url))      
    print('      Data length:    ', len(X2.data)) 
    Vectr1 = joblib.load(MODEL_PATH['MetaText']+'LSVC_TfIdfVectorizer.pkl')
    ts_sparse_data = Vectr1.transform(X2.data)
    X2_data = ts_sparse_data.toarray()   
    # X2_data =X2.data
    
    y2_pred       = Models[0].predict(X2_data)
    y2_pred_name  = [LABEL_NAMES[i] for i in y2_pred]
    if output_file is not None:
        output = open(output_file, 'w' ,newline ='', encoding='utf-8')
        output_csv = csv.writer(output ,quoting=csv.QUOTE_MINIMAL)
        for url, pred in zip(X2.url, y2_pred):
            output_csv.writerow(url.split('.')+ [LABEL_NAMES[pred]])
        output.close()
        
    print_results(X2, y2_pred, 'Meta Text')
    return
 


def classify_MetaData(MetaData, Models,output_file = None ):
    """
    --------------------------------------------------------------------------------
    Process Metadata Feature Data 
    --------------------------------------------------------------------------------
    """
    X3 = MetaData
    print('     ------------------------------------------------------------------------------')
    print('      Classify MetaData Feature set')
    print('     ------------------------------------------------------------------------------')    
    print('      URLS length:    ', len(X3.url))      
    print('      Data length:    ', len(X3.data))     
 
    vectrzr = joblib.load(MODEL_PATH['MetaData']+'RF_DictVectorizer.pkl')
    ts_sparse_data = vectrzr.transform(X3.data)
    ts_dense_data = ts_sparse_data.toarray()   
    
    scaler = joblib.load(MODEL_PATH['MetaData']+'RF_StandardScaler.pkl')
    X3_data = scaler.fit_transform(ts_dense_data)
    y3_pred = Models[0].predict(X3_data)

    # print(X3_data)
    # print(y3_pred)
    # print(len(X3.url), len(y3_pred_name), len(X3.label))
    
    print_results(X3, y3_pred, 'Meta Data')
    return

    
def print_results(X3,y3_pred, Title):
    """
    print results  
    """
    ctr = 0
    match = 0
    no_match = 0
    print('\n     ',Title,' Classfication Results: \n')
    print('     ------ URL ----------------------     ------ Predicted Label ------')
    for url, pred, orig_lbl  in zip(X3.url, y3_pred, X3.label):
        ctr +=1
        if orig_lbl is None:
            orig_lbl =  99
            orig_lbl_name =  '  -----  '
        else:
            orig_lbl_name = LABEL_NAMES[orig_lbl]
        print('     {:<35s}   {:2d} {:30s}'.format(url, pred, LABEL_NAMES[pred]))

        if pred == orig_lbl:
            match +=1
        else:
            no_match +=1 
    print()
    return
    
''' 
#--------------------------------------------------------------------------
#-- MAIN 
#--------------------------------------------------------------------------'''
def Classify(FeatureFiles, ModelFiles, classify_fs, output_file): 

    print(' --> Classification Started at:', 
                datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    output_path = REL_PATH + OUTPUT_PATH

    FILE_SFX = datetime.now().strftime("%m_%d_%Y@%H%M")
    print('     Classify: ', classify_fs)
    print('     Output will be written to  :', output_file)
    Models    = {}
    featureSet = {}
    start = time.time()

    print('\n     >> Model Search Paths:')   
    for i in MODEL_PATH:  
        print('       ',i, ' - :',MODEL_PATH[i])
        
    print('\n     >> FeatureSet datasets')
    for i in FeatureFiles:
        FeatureFiles[i] = INPUT_PATH+FeatureFiles[i] 
        print('        %s dataset: %s'%(i, FeatureFiles[i]))
        tss = TrainingSet()
        tss.load_from_Pkl(FeatureFiles[i])
        featureSet[i] = tss
            
    print('\n     >> Pretrained models')   
    for i in [ j for j in ModelFiles if len(ModelFiles[j]) != 0]:
        Models[i] = [joblib.load(MODEL_PATH[i]+j+'.pkl') for j in ModelFiles[i]]
            # print( 'model is ', j, ' path :' ,MODEL_PATH[i]+j+'.pkl')
        print('\n        ** ',i,'- Model files: ')
        for j in Models[i]:
            print('            >> ', j, '\n')
    

    '''
    classify the selected feature set
    '''
    if classify_fs in  ['htmltext', 'all']:
        classify_HtmlText(featureSet['HtmlText'], Models['HtmlText'], output_file)            
    
    if classify_fs in  ['metatext', 'all']:
        classify_MetaText(featureSet['MetaText'], Models['MetaText'])            
    
    if classify_fs in  ['metadata', 'all']:    
        classify_MetaData(featureSet['MetaData'], Models['MetaData'])            
    
    
    elapsed_time = time.time() - start
    print('     Elapsed time: %.2f seconds'%(elapsed_time))
    print(' --> Classification ended at:', 
                datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    print()
    return
    
    
#--------------------------------------------------------------------------
#  Main Driver 
#--------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification ')
    parser.add_argument('output_file', metavar='Output_filename',  
                        help='Output file in CSV format containing classified domain names: domainname, be, label')    
    parser.add_argument('-fs', metavar='Feature set',  choices=['htmltext', 'metatext', 'metadata', 'all'], 
                        default= 'htmltext', dest='classify_fs',
                        help='Feature set to classify can be one of the following: htmltext, metatext, or metadata')    
                     
    args = parser.parse_args()

    FeatureFiles = { 'HtmlText': 'TSTFS_HtmlText0511@2122.pkl',
                     'MetaData': 'TSTFS_MetaData0511@2122.pkl',    
                     'MetaText': 'TSTFS_MetaText0511@2122.pkl'}

    ModelFiles   = { 'HtmlText': ['LR_05_13_2017@1144'] ,
                     'MetaData': ['RF_05_12_2017@1731'],
                     'MetaText': ['LSVC_05_12_2017@1724']}
    
    Classify(FeatureFiles, ModelFiles, args.output_file, args.classify_fs)
    exit(' Classification completed successfully')

    