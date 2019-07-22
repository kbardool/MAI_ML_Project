'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
# from datetime                        import datetime
# import pickle, os, io, json, sys, time
import numpy as np
# import matplotlib.pyplot as plt
from Training                       import CLF_Pipeline, CLF_Pipeline1, ADB_Pipeline, split_data
from sklearn.model_selection         import StratifiedKFold
from sklearn.linear_model            import LogisticRegression
from sklearn.ensemble                import AdaBoostClassifier
from sklearn.tree                    import DecisionTreeClassifier
from sklearn.ensemble.forest         import RandomForestClassifier
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import SGDClassifier
from sklearn.svm.classes             import SVC
from sklearn.svm                     import LinearSVC
from sklearn.naive_bayes             import GaussianNB
from sklearn.preprocessing            import label_binarize
from sklearn.metrics                 import zero_one_loss
from common.LABELS                   import LABEL_NAMES, LABEL_CODES 

'''
#--------------------------------------------------------------------------
#-- LinearSVC Classifier setup  
#--------------------------------------------------------------------------'''  
def LSVC_classifier(X_all, Y_all, fsname, train_ratio = 0.8,
                     output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx       = 'LSVC_'
    CLF_name      = ' Linear SVC' 
    CLF_clf       = LinearSVC()
    
    CLF_parm_grid= {'C'             : [ 100.00] ,
                    'loss'          : ['squared_hinge'],   
                    'penalty'       : ['l2'],   
                    'dual'          : [False],   
                    'fit_intercept' : [True],   
                    'class_weight'  : ['balanced'],
                    'intercept_scaling' : [1.0],   
                    'verbose'       : [4],
                    'multi_class'   : ['ovr'],
                    'max_iter'      : [5000]                    
                    }                         
    # CLF_parm_grid= {'C'             : [ 10.0] ,
                    # 'loss'          : ['squared_hinge'],   
                    # 'penalty'       : ['l1'],   
                    # 'verbose'       : [4],
                    # 'max_iter'      : [10]                    
                    # }                         

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)     
          
    return CLF_bestclf, report 
   
 
'''
#--------------------------------------------------------------------------
#-- Logistical Regression Classifier setup  
#--------------------------------------------------------------------------'''    
def LR_classifier(X_all, Y_all, fsname, train_ratio = 0.8, 
                    output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx   = 'LR_'
    CLF_name  = 'Logistic Regression'+'-'+fsname
    CLF_clf   = LogisticRegression(random_state = 33)  
    # CLF_parm_grid = {'penalty'     : ['l2'], 
                    # 'C'            : [1, 10, 100],
                    # 'class_weight' : ['balanced'],
                    # 'max_iter'     : [5000],
                    # 'solver'       : ['newton-cg'],    #, 'lbfgs', 'sag'],
                    # 'multi_class'  : ['ovr'],          #,'multinomial'],
                    # 'warm_start'   : [True],
                    # 'verbose'      : [3]
                    # }
    CLF_parm_grid = {'penalty'     : ['l2'], 
                    'C'            : [1, 10, 100 ],
                    'class_weight' : ['balanced'],
                    'max_iter'     : [5000],
					# 'solver' : ['liblinear']
                    'solver' : ['newton-cg'],    #, 'lbfgs', 'sag'],
                    'multi_class'  : ['ovr', 'multinomial'],          #,'multinomial'],
                    'warm_start'   : [True],
                    'verbose'      : [3]
                    }

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)      
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)        
    return CLF_bestclf, report


'''
#--------------------------------------------------------------------------
#-- Gaussian Naive Bayes Classifier setup  
#--------------------------------------------------------------------------'''  
def GB_classifier(X_all, Y_all, fsname, train_ratio = 0.8,                   
                    output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx       = 'GB_'
    CLF_name      = 'Gaussian Naive Bayes'+'-'+fsname 
    CLF_clf       = GaussianNB()
    
    priors = np.bincount(Y_trn)
    CLF_parm_grid = { 
#                     'priors':priors,
                    }

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)            
    return CLF_bestclf, report
   
    
'''
#--------------------------------------------------------------------------
#-- Multinomial Bayes Classifier setup  
#--------------------------------------------------------------------------'''  
def MB_classifier(X_all, Y_all, fsname, train_ratio = 0.8,
                    output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx       = 'MB_'
    CLF_name      = 'MultiNomial Bayes'+'-'+fsname 
    CLF_clf       = MultinomialNB()
    CLF_parm_grid = { 
                    'fit_prior':[True],
                    'alpha':[ 0.0005 ]
                    # 'alpha':[ 0.1 ]
                   }   
    # CLF_parm_grid = { 
                    # 'fit_prior': [True, False], 
                    # 'alpha': [1, 0.1, 0.01, 0.005, 0.001, 0.0005]}

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
                   
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)        
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)             
    return CLF_bestclf, report

#--------------------------------------------------------------------------
#-- SVC Classifier setup  
#--------------------------------------------------------------------------'''  
def SVC_classifier(X_all, Y_all, fsname, train_ratio = 0.8,
                    output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx       = 'SVC_'
    CLF_name      = 'Support Vector Classification'+'-'+fsname 
    CLF_clf       = SVC(random_state=92)
    CLF_parm_grid = { 'C'       : [1.0, 10.0, 35.0],
                      'kernel'  : [ 'linear'],
                      'decision_function_shape' : ['ovr']
#                     'learning_rate':[ 'optimal' ]
                    }
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)    
    
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    return CLF_bestclf, report

    
'''
#--------------------------------------------------------------------------
#-- SGD Classifier setup  
#--------------------------------------------------------------------------'''  
def SGD_classifier(X_all, Y_all, fsname, train_ratio = 0.8,
                     output_path = None, TfIdf = True, NumFolds = 5):
    CLF_pfx       = 'SGD_'
    CLF_name      = 'Stochastic Gradient Descent'+'-'+fsname 
    CLF_clf       = SGDClassifier(random_state=42)
    CLF_parm_grid = { 'loss'   : ['hinge'],
                      'alpha'  : [ 0.1, 0.001 ,0.00001],
                      'penalty': ['l1','l2'],
                      'n_iter' : [5, 20, 50],
                      'fit_intercept': [True]
                   }
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
        
    if TfIdf:
        CLF_bestclf, report = CLF_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    return CLF_bestclf, report


