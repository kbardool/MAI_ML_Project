'''
Created on May 2, 2017

@author: kevin.bardool
'''

import itertools
import pickle, os, io, json
import sys, time
from itertools                        import cycle
from scipy                            import interp
from datetime                         import datetime
from imblearn.over_sampling           import SMOTE
from imblearn.under_sampling          import ClusterCentroids
from sklearn.preprocessing            import label_binarize
from sklearn.feature_selection.rfe    import RFECV
from sklearn                          import metrics
from sklearn.decomposition            import TruncatedSVD
from sklearn.ensemble                 import AdaBoostClassifier
from sklearn.externals                import joblib
from sklearn.feature_extraction       import DictVectorizer
from sklearn.feature_extraction.text  import TfidfVectorizer
# from sklearn.linear_model             import LogisticRegression
# from sklearn.linear_model             import SGDClassifier
# from sklearn.naive_bayes              import MultinomialNB
# from sklearn.ensemble.forest          import RandomForestClassifier
from sklearn.metrics                  import confusion_matrix, roc_curve,auc
from sklearn.model_selection          import GridSearchCV
from sklearn.model_selection          import StratifiedKFold
from sklearn.model_selection          import train_test_split

from sklearn.pipeline                 import Pipeline
from sklearn.svm                      import SVC
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.model_selection            import learning_curve
from sklearn.model_selection          import ShuffleSplit
# from classes.training_set            import TrainingSet
from common.LABELS                   import LABEL_NAMES, LABEL_CODES, BINARY_LABEL_NAMES
from common.utils                    import write_stdout, write_picklefile, write_stdout, write_strstream, write_bytestream

import matplotlib.pyplot             as plt
import numpy as np

'''
#--------------------------------------------------------------------------
#-- Vectorize Text Data
#--------------------------------------------------------------------------'''
def vectorize_TextData(trainingSet, max_df=0.8, min_df = 3, output = None): 
       
    print('\n\n --> Vectorize textual features')    
    vec = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None,
                         max_features=10000,
                         max_df = max_df, 
                         min_df = min_df, 
                         use_idf =True, 
                         smooth_idf=True,
                         norm='l2')
    ts_sparse_data = vec.fit_transform(trainingSet.data)
    ftr_names = vec.get_feature_names()
    vec_parms = vec.get_params(deep=True) 
    print('     Number of Features      : ', len(ftr_names), ftr_names[:5])
    print('     Dict vectorization parms: ', vec_parms)
    print('     Shape of data matrix is : ', ts_sparse_data.shape)   
    if output is not None:
        joblib.dump(vec  , output)    
    return ts_sparse_data, ftr_names, vec_parms 

    '''
#--------------------------------------------------------------------------
#-- Vectorize Dictionary   
#--------------------------------------------------------------------------'''
def vectorize_Dict(featureSet): 
    print(' --> Vectorize features')    
    vec = DictVectorizer()
    ts_data = vec.fit_transform(featureSet.data)
    ftr_names = vec.get_feature_names()
    vec_parms = vec.get_params(deep=True) 
    return ts_data, vec 

'''
#--------------------------------------------------------------------------
# Split data into test and training   
#--------------------------------------------------------------------------'''
def split_data(X_all, Y_all, train_split): 
    print(' --> Split into training and test')
    X_train, X_test, Y_train, Y_test = train_test_split(
                X_all, Y_all, train_size=train_split, random_state=33)
    # print('    X_train:',len(X_train),' Y_train:',len(Y_train)) 
    return X_train, Y_train, X_test, Y_test
    
'''
#--------------------------------------------------------------------------
#-- Product post training report
#--------------------------------------------------------------------------'''
def training_report(CLF_gs, CLF_name, X_train, Y_train, X_test, Y_test):

    print('--- ',CLF_name, ' Report  ----')
    print('  Best Param Set       : ', CLF_gs.best_params_)
    print('  Best Score found     : ', CLF_gs.best_score_)
    print('\n\n--- Classifier Report ----------------------------------------------')
    print('  Accuracy (Score): ', CLF_gs.best_estimator_.score(X_test, Y_test)) 
    
    CLF_predicted = CLF_gs.best_estimator_.predict(X_test)          
    print(metrics.classification_report(Y_test, 
                                        CLF_predicted,
                                        labels=LABEL_CODES,
                                        target_names=LABEL_NAMES))    
                                        
                                        
    cm = metrics.confusion_matrix(Y_test, CLF_predicted)
    print("Confusion matrix:")
    print(cm)    
    print()
    plot_confusion_matrix(cm, LABEL_NAMES,
                          normalize=False,
                          title='Confusion matrix '+CLF_name,
                          cmap=plt.cm.Blues)
    plot_roc_curve(CLF_gs.best_estimator_, CLF_name, X_test, Y_test)
    
    # print(' Original Data shape     : ', len(X_train) , ' Original Label shape    : ', len(X_test))
    # print(' Oversampled Data shape  : ', len(Y_train) , ' Oversampled Label shape : ',len(Y_test))
    # too time consuming to compute. have commented it out for now
    plot_learning_curve(CLF_gs.best_estimator_, 
                        CLF_name, np.vstack((X_train, X_test)), np.hstack((Y_train,Y_test)), ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))    
    
    print('\n --> End ',CLF_name,' Classifier at: ' ,datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))      
    return

      
    
'''
#--------------------------------------------------------------------------
#-- write model and results to pickle files
#--------------------------------------------------------------------------'''
def persist_model(CLF_gs, output_path, CLF_pfx, FILE_SFX):
    
    joblib.dump(CLF_gs.best_estimator_  , output_path+CLF_pfx+FILE_SFX+'.pkl')
    
    write_picklefile(output_path, CLF_pfx+FILE_SFX+'_RES',
                { 'cv_results_' : CLF_gs.cv_results_ ,
                  'best_index_' : CLF_gs.best_index_ , 
                  'best_score_' : CLF_gs.best_score_ , 
                  'best_params_': CLF_gs.best_params_ ,
                  'n_splits'    : CLF_gs.n_splits_     })
    print(' Best Model written to :', output_path+CLF_pfx+FILE_SFX+'.pkl') 
    return
    

      
'''
#--------------------------------------------------------------------------
#-- Use SMOTE alg to oversample  data
#--------------------------------------------------------------------------'''    
def  smote_data(X_data, Y_data, times, kind='regular'):    
    X_tr = X_data
    Y_tr = Y_data
    print('  Original Undersampled dataset label counts {}'.format(np.bincount(Y_data)))
    
    # possible kinds are:  ‘regular’, ‘borderline1’, ‘borderline2’, ‘svm’.
    
    sm = SMOTE(ratio='auto'  , 
               random_state=12,
               k_neighbors=5,
               m_neighbors=10,
               out_step=0.5, kind=kind,
               svm_estimator=SVC())
    
    for i in range(times):
        X_tr, Y_tr = sm.fit_sample(X_tr, Y_tr)
#         print('  SMOTE iteration {} dataset label counts {}'.format(i, np.bincount(Y_tr)))
    
    print('\n\n  Final Undersampled dataset label counts {}'.format(np.bincount(Y_tr)))    
    return X_tr, Y_tr                

'''
#--------------------------------------------------------------------------
#-- Pipeline Setup and Execution  
#--------------------------------------------------------------------------'''    
def CLF_Pipeline(X_train, Y_train, X_test, Y_test, 
                 CLF_clf, CLF_clfparms, CLF_name, CLF_pfx, 
                 output_path, NumFolds=5):
           
    save_stdout = sys.stdout                 
    sys.stdout  = io.StringIO()    
    
    FILE_SFX = datetime.now().strftime("%m_%d_%Y@%H%M")
    
    print('\n\n --> Start ',CLF_name,' Classifier (with TfIdf) at: ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    print(' Number of folds:', NumFolds)
    print(' CLF parm grid  :', CLF_clfparms)
    
    CLF_tfidf = TfidfVectorizer( strip_accents=None, lowercase=False  , preprocessor= None)
    CLF_cv    = StratifiedKFold( n_splits=NumFolds , random_state=None, shuffle=True)       
    CLF_ppl   = Pipeline([('tfidf' , CLF_tfidf),
                        ('clf'   , CLF_clf)])
    
    print(CLF_cv.get_n_splits())
    print(CLF_cv)
    
    # best hyper-parms from previous grid search
    CLF_tfidf_parms= {'tfidf__max_features':[10000],
                    'tfidf__max_df'    :[0.8],
                    'tfidf__min_df'    :[3],
                    'tfidf__use_idf'   :[True], 
                    'tfidf__smooth_idf':[True],
                    'tfidf__norm'      :['l2']}
    
    CLF_estparms    = {}
    print(' CLF_clfparm input:',CLF_clfparms)
    for i in CLF_clfparms:
        CLF_estparms['clf__'+i] = CLF_clfparms[i]
        
    # CLF_tfidf_parms= {'tfidf__max_features' : [1000, 5000],
                     # 'tfidf__max_df'        : [0.8, 0.9, 1.0],
                     # 'tfidf__min_df'        : [  2,   4,   6],
                     # 'tfidf__use_idf'       : [False, True], 
                     # 'tfidf__smooth_idf'    : [False,True],
                     # 'tfidf__norm'          : ['l1','l2']}
    
    CLF_parm_grid = {**CLF_tfidf_parms ,**CLF_estparms}
    
    print(' CLF parm grid:',CLF_clfparms)
    print(' Total parm grid:',CLF_parm_grid)

    CLF_gs = GridSearchCV(CLF_ppl, 
                         param_grid=CLF_parm_grid, 
                         scoring='accuracy',
                         cv = CLF_cv,
                         verbose=2,
                         n_jobs=4)      

    start     = time.time()     
    CLF_gs = CLF_gs.fit(X_train, Y_train)
    elapsed_time  = time.time() - start
    
    print('\n\n  %s Training ended - Elapsed time: %.2f seconds'%(CLF_name, elapsed_time))

    training_report(CLF_gs, CLF_name, X_train, Y_train, X_test, Y_test)
    persist_model(CLF_gs, output_path, CLF_pfx, FILE_SFX)
    report = sys.stdout.getvalue()
    write_stdout(output_path, CLF_pfx+FILE_SFX, sys.stdout )     
    
    sys.stdout = save_stdout
    print(report)
    return CLF_gs.best_estimator_ ,report

'''
#--------------------------------------------------------------------------
#-- Pipeline Setup and Execution  - No TFIDF
#--------------------------------------------------------------------------'''    
def CLF_Pipeline1(X_train, Y_train, X_test, Y_test, 
                 CLF_clf, CLF_clfparms, CLF_name, CLF_pfx,
                 output_path, NumFolds=5):
           
    save_stdout = sys.stdout                 
    sys.stdout  = io.StringIO()    
    
    FILE_SFX = datetime.now().strftime("%m_%d_%Y@%H%M")
   
    print('\n\n --> Start ',CLF_name,' Classifier at: ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))

    CLF_cv    = StratifiedKFold(n_splits=NumFolds, random_state=None, shuffle=True)       
    CLF_ppl = Pipeline([('clf'   , CLF_clf)])
    
    # best hyper-parms from previous grid search
    CLF_estparms    = {}
    for i in CLF_clfparms:
        CLF_estparms['clf__'+i] = CLF_clfparms[i]
        
    CLF_parm_grid = {**CLF_estparms}
    
    print('\n  ',CLF_cv)
    print(' Total parm grid:',CLF_parm_grid)

    CLF_gs = GridSearchCV(CLF_ppl, 
                         param_grid=CLF_parm_grid, 
                         scoring='accuracy',
                         cv = CLF_cv,
                         verbose=4,
                         n_jobs=4)      

    start     = time.time()     
    CLF_gs = CLF_gs.fit(X_train, Y_train)
    elapsed_time  = time.time() - start
    
    print('\n\n  %s Training ended - Elapsed time: %.2f seconds'%(CLF_name, elapsed_time))

    training_report(CLF_gs, CLF_name, X_train, Y_train, X_test, Y_test)
    persist_model(CLF_gs, output_path, CLF_pfx, FILE_SFX)
    report = sys.stdout.getvalue()
    write_stdout(output_path, CLF_pfx+FILE_SFX, sys.stdout )  

   
    
    sys.stdout = save_stdout
    print(report)
    return CLF_gs.best_estimator_ ,report

'''
#--------------------------------------------------------------------------
#--Adaboost Pipeline  
#--------------------------------------------------------------------------    
'''
def ADB_Pipeline(X_train, Y_train, X_test, Y_test, 
                       CLF_clf, CLF_clfparms, CLF_name, CLF_pfx,
                       output_path , NumFolds=5):
    save_stdout = sys.stdout                 
    sys.stdout  = io.StringIO()    

    CLF_pfx  = 'ADA'+CLF_pfx
    CLF_name = 'Adaboost - '+CLF_name

    FILE_SFX = datetime.now().strftime("%m_%d_%Y@%H%M")

    print(' --> Start ',CLF_name,' Classifier at: ',
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))

    CLF_estparms    = {}
    print(' CLF_clfparm input:',CLF_clfparms)
    for i in CLF_clfparms:
        CLF_estparms['clf__base_estimator__'+i] = CLF_clfparms[i]
        
    print('CLF_estparms: ',CLF_estparms)
       
    CLF_cv    = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)       
    ADA_clf   = AdaBoostClassifier(base_estimator = CLF_clf)

    CLF_adaparms = {'clf__learning_rate': [0.1, 0.05],    # , 0.1],
                    'clf__n_estimators' : [400],   # , 400],
                    'clf__algorithm'    : ['SAMME']
                   }    

    CLF_ppl = Pipeline([('clf'   , ADA_clf)])   

    CLF_parm_grid = {**CLF_adaparms, **CLF_estparms}   

    print(' Total parm grid:',CLF_parm_grid)
    
    CLF_gs = GridSearchCV(CLF_ppl, 
                         param_grid=CLF_parm_grid, 
                         scoring='accuracy',
                         cv = CLF_cv,
                         verbose=4,
                         n_jobs=4)      

    start         = time.time()     
    CLF_gs        = CLF_gs.fit(X_train, Y_train)
    elapsed_time  = time.time() - start
    
    
    print('\n\n  %s Training ended - Elapsed time: %.2f seconds'%(CLF_name, elapsed_time))

    training_report(CLF_gs, CLF_name, X_train, Y_train, X_test, Y_test)
    persist_model(CLF_gs, output_path, CLF_pfx, FILE_SFX)
    report = sys.stdout.getvalue()
    write_stdout(output_path, CLF_pfx+FILE_SFX, sys.stdout )     
    
    sys.stdout = save_stdout
    print(report)
    return CLF_gs.best_estimator_ ,report

    
    
'''
#--------------------------------------------------------------------------
#  Singular Value Decomposition Process  
#--------------------------------------------------------------------------'''
def SVD_process(featureSet):      
    svd = TruncatedSVD(n_components=10,n_iter= 20,random_state=42)
    svd.fit(featureSet)
    exp_var_ratio = svd.explained_variance_ratio_
#     print(' PCA components:', svd.components_)
    print(' PCA n_components:', svd.explained_variance_)
    print(' PCA explained variance ratio:', exp_var_ratio)
    
    for i in range(len(exp_var_ratio)):
        print(exp_var_ratio[:i].sum())
    
    reduced_fs = svd.transform(featureSet)
    print(reduced_fs.shape)
    return reduced_fs



'''
#--------------------------------------------------------------------------
#-- Build Roc 
# Original Source code from :
#   https://devdocs.io/scikit_learn/auto_examples/model_selection/plot_roc#sphx-glr-auto-examples-model-selection-plot-roc-py
# with customization applied for or project
#--------------------------------------------------------------------------'''
def plot_roc_curve(CLF_clf, CLF_name, X_test, y_test):  
  
    y_test  = label_binarize(y_test, LABEL_CODES)
            
    try:
        y_score = CLF_clf.decision_function(X_test)
    except AttributeError as e:
        try:
            y_score = CLF_clf.predict_proba(X_test)
        except AttributeError as e:
            print("   Model has no decision_function, roc_curve cannot be plotted ", e)
            return
            
            
    n_classes = 11
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
 
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i] )
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


 
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua'    , 'darkorange', 'cornflowerblue', 'teal'  , 'indigo', 
                    'seagreen', 'navy'      , 'turquoise'     , 'yellow', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class: <{0}> (area = {1:0.2f})'
                 ''.format(LABEL_NAMES[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - '+CLF_name)
    plt.legend(loc="lower right")
    plt.show()

'''
#--------------------------------------------------------------------------
#-- Plot Learning Curve
#   Original Source code from :
#   https://devdocs.io/scikit_learn/auto_examples/model_selection/plot_learning_curve#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
#    
#--------------------------------------------------------------------------'''            
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title('Test/Training Learning Curve - '+ title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean  + test_scores_std , alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return 

'''
#--------------------------------------------------------------------------
#-- Plot Confusion Matrix
#   Original Source code from :
#       http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
#   with applied customization for our project
#--------------------------------------------------------------------------'''        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return
    
'''
#--------------------------------------------------------------------------
#-- Recursive Feature Selection  
#--------------------------------------------------------------------------'''
def featureSelection(  X, Y):
    print(' --> Feature Selection')
    SVC_cv  = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    SVC_clf = SVC(verbose=True,kernel='linear')
    selector= RFECV(SVC_clf, step=1, cv=SVC_cv)
    selector = selector.fit(X, Y)
    
    print('n_features_ :', selector.n_features_)
    print('support_    :', selector.support_)
    print('ranking_    :', selector.ranking_)
    print('grid_scores_:', selector.grid_scores_)
    print('estimator_  :', selector.estimator_m)
    
    return 

'''
#--------------------------------------------------------------------------
#-- Recursive Feature Selection  
#--------------------------------------------------------------------------'''
def featureSelection1(X, Y):
    print(' --> Feature Selection')
    DT_cv  = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    DT_clf = DecisionTreeClassifier()
    selector= RFECV(DT_clf, step=1, cv=DT_cv)
    selector = selector.fit(X, Y)
    
    print('n_features_ :', selector.n_features_)
    print('support_    :', selector.support_)
    print('ranking_    :', selector.ranking_)
    print('grid_scores_:', selector.grid_scores_)
    print('estimator_  :', selector.estimator_)
    
    return 

'''
#--------------------------------------------------------------------------
#-- AdaBoost  with Decision Tree
#--------------------------------------------------------------------------'''    
def DT_AdaBoost_Compare(X_train, y_train, X_test, Y_test, output_path):
    n_estimators = 400
    learning_rate = 1.0

    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, Y_test)
    
    dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    dt_err = 1.0 - dt.score(X_test, Y_test)
    

    ada_discrete = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME")
    ada_discrete.fit(X_train, y_train)
    
    ada_real = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME.R")
    ada_real.fit(X_train, y_train)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
            label='Decision Stump Error')
    ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
            label='Decision Tree Error')
    
    ada_discrete_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
        ada_discrete_err[i] = zero_one_loss(y_pred, Y_test)
    
    ada_discrete_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
        ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)
    
    ada_real_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
        ada_real_err[i] = zero_one_loss(y_pred, Y_test)
    
    ada_real_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
        ada_real_err_train[i] = zero_one_loss(y_pred, y_train)
    
    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
            label='Discrete AdaBoost Test Error',
            color='red')
    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
            label='Discrete AdaBoost Train Error',
            color='blue')
    ax.plot(np.arange(n_estimators) + 1, ada_real_err,
            label='Real AdaBoost Test Error',
            color='orange')
    ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
            label='Real AdaBoost Train Error',
            color='green')
    
    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')
    
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    
    plt.show()
    
