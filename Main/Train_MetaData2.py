'''
Created on Apr 12, 2017

@author: kevin.bardool
'''

print('train metadata2 __name__ is ',__name__)
import numpy as np
import matplotlib.pyplot as plt

from Training                        import CLF_Pipeline1, ADB_Pipeline, split_data
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
#-- Logistical Regression Classifier setup  
#--------------------------------------------------------------------------'''    
def LR_classifier(X_all, Y_all, train_ratio = 0.8,
                    output_path = None, NumFolds = 5,  AdaBoost = False):
    CLF_pfx   = 'LR_'
    CLF_name  = 'Logistic Regression (MetaData)'
    CLF_clf   = LogisticRegression(random_state = 33)  
    CLF_parm_grid = {'penalty'     : ['l2'], 
                    'C'            : [5,  10,  50, 100],
                    'max_iter'     : [ 5000 ],
                    'solver'       : ['newton-cg'],
                    'multi_class'  : ['ovr', 'multinomial'],
                    'class_weight' : ['balanced'],
                    'warm_start'   : [True]}    
#     CLF_parm_grid = {   'penalty'  : ['l2'], 
#                     'C'            : [5],
#                     'class_weight' : ['balanced']} 

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , \
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    return CLF_bestclf, report


'''
#--------------------------------------------------------------------------
#-- Decision Tree Classifier setup  
#--------------------------------------------------------------------------'''  
def DT_classifier(X_all, Y_all, train_ratio = 0.8, 
                    output_path = None, NumFolds = 5, AdaBoost=False):
    CLF_pfx       = 'DT_'
    CLF_name      = 'Decision Tree (MetaData)' 
    CLF_clf       =  DecisionTreeClassifier()
# sel for adaboost     
    CLF_parm_grid = {'criterion': ['entropy'],
                    'splitter'  : ['random', 'best'],
                    'class_weight':[ 'balanced']
                    }        
    # CLF_parm_grid = {'criterion':['gini', 'entropy'],
                    # 'splitter' :['random', 'best'],
                    # 'class_weight':[ 'balanced', None ]
                    # }    
    
    # CLF_parm_grid = { 
                    # 'criterion'        : ['gini'],
                    # 'splitter'         : ['best'],
                    # 'max_features'     : [None],
                    # 'max_depth'        : [None],
                    # 'min_samples_split': [2],
                    # 'min_samples_leaf' : [1],
                    # 'class_weight'     : ['balanced']
                   # }
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    
    return CLF_bestclf, report


'''
#--------------------------------------------------------------------------
#-- Random Forest Classifier setup  
#--------------------------------------------------------------------------'''  
def RF_classifier(X_all, Y_all, train_ratio = 0.8, 
                    output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'RF_'
    CLF_name      = 'Random Forest (MetaData)' 
    CLF_clf       = RandomForestClassifier(verbose=3)
#     CLF_parm_grid = {'n_estimators': [20, 30],
#                      'criterion'   : ['entropy'],
#                      'max_features': [None],
#                      'oob_score'   : [False],
#                      'class_weight': ['balanced']
#                    }
    
    CLF_parm_grid = { 
                    'n_estimators'     : [50],
                    'criterion'        : ['entropy'],
                    'max_features'     : [None],
                    'max_depth'        : [None],
                    'min_samples_split': [2],
                    'min_samples_leaf' : [1],
                    'min_weight_fraction_leaf' : [0],
                    'max_leaf_nodes'   : [None],
                    'bootstrap'        : [True],
                    'class_weight'     : ['balanced']
                   }
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)          
    return CLF_bestclf, report

    
'''
#--------------------------------------------------------------------------
#-- LinearSVC Classifier setup  
#--------------------------------------------------------------------------'''  
def LSVC_classifier(X_all, Y_all, train_ratio = 0.8,
                     output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'LSVC_'
    CLF_name      = ' Linear SVC (MetaData)' 
    CLF_clf       = LinearSVC(random_state=92)
    
    CLF_parm_grid= {'C'             : [ 100.00, 200.00, 300.00] ,
                    'loss'          : ['squared_hinge'],   
                    'penalty'       : ['l2'],   
                    'dual'          : [False],   
                    'fit_intercept' : [True],   
                    'class_weight'  : [None,'balanced'],
                    'intercept_scaling' : [1.0],   
                    'verbose'       : [2],
                    'multi_class'   : ['ovr'],
                    'max_iter'      : [25000]                    
                    }                         
    # CLF_parm_grid= {'C'             : [ 10.0] ,
                    # 'loss'          : ['squared_hinge'],   
                    # 'penalty'       : ['l1'],   
                    # 'verbose'       : [4],
                    # 'max_iter'      : [10]                    
                    # }                         

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)     
          
    return CLF_bestclf, report 
   

'''
#--------------------------------------------------------------------------
#-- SVC Classifier setup  
#--------------------------------------------------------------------------'''  
def SVC_classifier(X_all, Y_all, train_ratio = 0.8,
                     output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'SVC_'
    CLF_name      = 'Support Vector Classification (MetaData)' 
    CLF_clf       = SVC(random_state=92)
            
    # CLF_parm_grid={'C':[0.01, 0.1,  10 ],
                    # 'kernel' :[ 'linear',   'rbf' ],
                    # 'gamma':[0.001, 0.005, 0.01, 0.05 ],
                    # 'decision_function_shape':['ovr']
                    # }
#     CLF_parm_grid = { 'C'       : [1.0, 10.0, 35.0],
#                       'kernel'  : [ 'linear'],
#                     }               
     
    CLF_parm_grid= {'C'             : [ 100, 200] ,
                    'kernel'        : ['linear'],   #'rbf', 'sigmoid', 'poly'],
                    'gamma'         : [0.001, 0.01, 0.1 ],         
                    'degree'        : [3],
                    'cache_size'    : [300.0],
                    'verbose'       : [True],
                    'class_weight'  : ['balanced'],
                    # 'decision_function_shape' : ['ovr'] 
                    }                         
                    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
        
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)     
          
    return CLF_bestclf , report
   


'''
#--------------------------------------------------------------------------
#-- Gaussian Naive Bayes Classifier setup  
#--------------------------------------------------------------------------'''  
def GB_classifier(X_all, Y_all, train_ratio = 0.8,
                    output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'GB_'
    CLF_name      = 'Gaussian Naive Bayes (MetaData)' 
    CLF_clf       = GaussianNB()
    CLF_parm_grid = { 
#                     'priors':priors,
                    }
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)            
    return CLF_bestclf, report
   

'''
#--------------------------------------------------------------------------
#-- Multinomial Bayes Classifier setup  
#--------------------------------------------------------------------------'''  
def MB_classifier(X_all, Y_all, train_ratio = 0.8,
                    output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'MB_'
    CLF_name      = 'MultiNomial Bayes (MetaData)' 
    CLF_clf       = MultinomialNB()
    CLF_parm_grid = { 
                    'fit_prior':[True,False],
                    'alpha':[1.0, 0.8, 0.6]
#                   'penalty':['l1','l2'],
#                   'learning_rate':[ 'optimal' ]
                   }

    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
                       
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)            
    return CLF_bestclf, report


'''
#--------------------------------------------------------------------------
#-- SGD Classifier setup  
#--------------------------------------------------------------------------'''  
def SGD_classifier(X_all, Y_all, train_ratio = 0.8,
                     output_path = None, NumFolds = 5, AdaBoost = False):
    CLF_pfx       = 'SGD_'
    CLF_name      = 'Stochastic Gradient Descent (SGD) (MetaData)' 
    CLF_clf       = SGDClassifier(verbose=True, random_state=None)
    CLF_parm_grid = {'loss'     : ['hinge'],
                     'penalty'  : ['L2'],
                     'alpha'    : [1e-3],
                     'n_iter'   : [ 5 ]
                    }
                    
                    
    # penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
    # alpha : float
    # learning_rate : 'constant’: eta = eta0, ‘optimal’: eta = 1.0 / (alpha * (t + t0)) [default], ‘invscaling’: eta = eta0 / pow(t, power_t)
    # power_t : double, optional
    
    
    print(' Labels Count :  ',np.bincount(Y_all))
    X_trn, Y_trn, X_tst, Y_tst = split_data(X_all, Y_all, train_ratio)
    
    if AdaBoost:
        CLF_bestclf, report = ADB_Pipeline(X_trn, Y_trn, X_tst, Y_tst , 
                CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)    
    else:
        CLF_bestclf, report = CLF_Pipeline1(X_trn, Y_trn, X_tst, Y_tst , 
                 CLF_clf, CLF_parm_grid, CLF_name, CLF_pfx, output_path, NumFolds)         
    return CLF_bestclf, report
