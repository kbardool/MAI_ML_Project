'''
Created on Apr 6, 2017

@author: kevin.bardool
'''

# import re , uuid, requests
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
# import io, bs4, bson, zlib , sys, re, urllib,unicodedata
import csv
import pandas as pd
from datetime import datetime   
from common.utils               import write_picklefile
 
# from models.stores.store import Store

__author__ = 'KBardool'


class TrainingSet(object):
    ts_header = ['URL','LABEL','LABEL_CD','DATA_LANG']
    def __init__(self):
        self.url            = []
        self.label          = []
        self.data_lang      = []  
        self.data           = []
        return
    
#--------------------------------------------------------------------------
#--   append training instance to Training Set
#--------------------------------------------------------------------------     
    def addTrainingInstance(self, trnFeature) :
        self.url.append(trnFeature.url)
        self.label.append(trnFeature.label_cd)
        self.data_lang.append(trnFeature.data_lang)
        self.data.append(trnFeature.data) 
#         print('fSet size:',len(self.url), len(self.label), len(self.data) )
        return

#--------------------------------------------------------------------------
#  Remove low frequency values 
#--------------------------------------------------------------------------
    def postProcess(self):
        df1 = pd.DataFrame(self.data)
        df1.insert(0,'url',self.url)
        df1.insert(1,'label',self.label)
        df1.set_index(['url','label'],inplace=True)
        rows,cols = df1.shape
        print(' original data frame has ',rows, 'rows and ',cols,' columns')
        # delete 
        xs = df1.sum(axis=0)
        xsn = xs[xs <= 10].index.tolist()
        df1 = df1.drop(xsn, axis = 1)
        
        xs = df1.count(axis=0)
        xsn = xs[xs <= 10].index.tolist()
        df1 = df1.drop(xsn, axis = 1)    
        
        rows,cols = df1.shape
        print(' pruned data frame has ',rows, 'rows and ',cols,' columns')
        
        trSet = TrainingSet()
        for i in df1.iterrows():
            trSet.url.append(i[0][0])
            trSet.label.append(i[0][1])
            trSet.data_lang.append('na')
            trSet.data.append(i[1].dropna().to_dict())
            
        print(' url  : ', trSet.url[:10])
        print(' label: ', trSet.label[:10])
        print(' Data  : ', trSet.data[:10]) 
        return trSet
              
#--------------------------------------------------------------------------
#--   write Complete Feature Set data to pickle file 
#--------------------------------------------------------------------------    
    def write_to_Pkl(self, name , output_path, file_sfx):    
        write_picklefile(output_path, name+file_sfx,
                [ self.url, self.label, self.data_lang, self.data ])
        print('     wrote html_text pickled output to:',name+file_sfx,'.pkl')

#         print('fSet size:',len(self.url), len(self.label), len(self.data) )      
#         print(' Meta kwrds url  : \t len:  ', len(self.url)  , self.url[:10])    
#         print(' Meta kwrds label: \t len:  ', len(self.label), self.label[:10])    
#         print(' Meta kwrds lang : \t len:  ', len(self.data_lang)  , self.data_lang[:10])    
#         print(' Meta kwrds data : \t len:  ', len(self.data)  , self.data[:10]) 

        return

#--------------------------------------------------------------------------
#--   write Complete Feature Set data to pickle file 
#--------------------------------------------------------------------------    
    def load_from_Pkl(self, file):    
        try:
            f_obj = open(file, 'rb')      
            data = pickle.load(f_obj)
        except Exception as e:
            print('@@@@@ An error occurred while writing feature set to picklefile: ',file ,'  ',e)
        else:
            self.url   = data[0]
            self.label = data[1]
            self.data_lang = data[2]
            self.data   = data[3]
        return
        
        
#--------------------------------------------------------------------------
#--   write Complete Feature Set data to pickle file 
#--------------------------------------------------------------------------     
    def write_to_CSV(self, name, output_path, file_sfx):    
        with open(output_path+ name+file_sfx+'.csv', 'w' ,newline ='', encoding='utf-8') as f_obj:
            csv_writer = csv.writer(f_obj ,quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['URL','LABEL_CD','DATA_LANG','DATA_LEN'])     
            for a,b,c,d in zip(self.url,self.label,self.data_lang, self.data):
                csv_writer.writerow([a,b,c,len(d),d])
            f_obj.close()        
        print('     CSV output written to:',name+file_sfx,'.pkl')
        return 

        
#--------------------------------------------------------------------------
#--   write Complete Feature Set data to pickle file 
#-------------------------------------------------------------------------- 
    def write_to_CSV2(self, name, output_path, file_sfx):    
        with open(output_path+ name+file_sfx+'.csv', 'w' ,newline ='', encoding='utf-8') as f_obj:
            csv_writer = csv.writer(f_obj ,quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['URL','LABEL_CD','DATA_LANG','DATA_LEN'])
            if type(self.data[0]) == type({}):
                cols = [ i for i in self.data[0]]
                print(len(cols),'   ',cols)
            else:
                print(type(self.data[0]))
                
            for a,b,c,d in zip(self.url,self.label,self.data_lang, self.data):
                csv_writer.writerow([a,b,c,len(d),d])
            f_obj.close()        
        # print('     CSV output written to:',name+file_sfx,'.pkl')
        return 

        return

 
#--------------------------------------------------------------------------
#  get <link> realated information 
#--------------------------------------------------------------------------
           
    def to_json(self):
#         self.get_page_metadata_info()
        return {
#             "_id"           :   self._id,
            "url"           :   self.url,
            "label"         :   self.label,
            "data_lang"     :   self.meta_keywords,
            "tag_counts"    :   self.tag_counts,     
            "meta_data"     :   self.meta_data,       
            'timestamp'     :   datetime.now()            
        }
    
    
#--------------------------------------------------------------------------
#--   append a training Feature to it's Feature Set collection 
#-------------------------------------------------------------------------- 
    def save_to_db(self):
#         pagefeaturesTbl = MongoPageFeatures()
#         pagefeaturesTbl[self.url] = self.to_json()
        return 
    
    def __repr__(self):
#         return "<Item {} with URL {} Page Title {}>".format(self._id, self.url, self.html_title)
        return
                
        self.label         = []
        self.data_lang      = []  
        self.data           = []
        return
    
    
    @classmethod
    def get_by_id(cls, _id):
#         item_data = Database.find_one("pagefeatures", {"_id": _id})
#         return cls(**item_data)
        return