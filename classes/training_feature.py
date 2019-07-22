'''
Created on Apr 6, 2017

@author: kevin.bardool
'''

# import re , uuid, requests
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import  csv, re, string
import  common.utils as utils
from datetime                   import datetime   
from common.database            import Database
from classes.mng_pgfeatures     import MongoPageFeatures
from classes.mng_trnfeatures    import MongoTrnFeatures
from nltk.tokenize              import word_tokenize
from nltk.corpus                import stopwords


__author__ = 'KBardool'

#--------------------------------------------------------------------------
#  Training Feature Class
#--------------------------------------------------------------------------
class TrainingFeature(object):
    NLTK_LANGS = {'nl':'dutch' , 'en':'english', 'fr':'french',
                  'de':'german', 'da':'danish' , '??':'dutch'}
    
    # Some strings for ctype-style character classification
    # whitespace = ' \t\n\r\v\f'
    # ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    # ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # ascii_letters = ascii_lowercase + ascii_uppercase
    # digits = '0123456789'
    # hexdigits = digits + 'abcdef' + 'ABCDEF'
    # octdigits = '01234567'
    # punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    # printable = digits + ascii_letters + punctuation + whitespace   
    # unicd_punct  = re.compile(r'[\u2018\u2019]')
    # ascii_cc     = re.compile(r'[\x01-\x1F\x7F]')    
    
    html_tags  = '<[a-zA-Z!\/][^>]*>' 
    wtspc_qt   = '[\s\"]+'
    non_alphanum = re.compile(r'[^%s]'% re.escape(string.digits + string.ascii_letters ))    
    punct_ws     = re.compile('[%s]+' % re.escape(string.punctuation))
    ws_qt        = re.compile('%s' % wtspc_qt)
    rem_www      = re.compile(r'www.')
    embded_html  = re.compile(html_tags)   
    unicd_cc   = re.compile(r'[\u2018\u2019\u0000-\u0008\u000a-\u000c\u000e-\u001f\ufffe-\uffff\ue000-\uf8ff]')

#--------------------------------------------------------------------------
#  generate html text feature  
#--------------------------------------------------------------------------    
    def __init__(self, featureName, pageFeature,  debug = False, dbgfile=None, path = None):

        self.debug          = debug
        self.dbgfile        = dbgfile
        self.output_path    = path
        
        if dbgfile is not None:
            self.csv_writer     = csv.writer(self.dbgfile ,quoting=csv.QUOTE_MINIMAL)
        self.url            = pageFeature.url
        self.label          = pageFeature.label
        self.label_cd       = pageFeature.label_cd
        self.html_lang      = pageFeature.html_lang
        self.data_lang      = '??'
        self.data           = None
        self.feature_name   = featureName
        return
    
    
#--------------------------------------------------------------------------
#  generate html text feature  
#--------------------------------------------------------------------------
    def as_htmltext(self, pageFeatures):
        content = pageFeatures.html_text
        stmd_list, stmd_text = self.process_text_content(content,pageFeatures.text_lang)
        self.data_lang = pageFeatures.text_lang
        self.data = stmd_text
        return stmd_list , stmd_text


#--------------------------------------------------------------------------
#  generate meta keywords feature 
#--------------------------------------------------------------------------
    def as_metatext(self, pageFeatures):
        content = ' '.join((pageFeatures.meta_text , pageFeatures.html_title))   
        stmd_list, stmd_text = self.process_text_content(content,pageFeatures.text_lang)
        self.data_lang = pageFeatures.text_lang
        self.data = stmd_text
        return stmd_list, stmd_text
    
    
#--------------------------------------------------------------------------
#  get <link> realated information 
#--------------------------------------------------------------------------
    def as_metadata(self, pageFeatures ):
        self.data = {**pageFeatures.meta_data }
        
        if 200<= self.data['code'] <= 300:
            self.data['code'] = 1
        else:
            self.data['code'] = 0
        return 

#--------------------------------------------------------------------------
#  get img information
#-------------------------------------------------------------------------- 
    @staticmethod
    def process_text_content(inp_text, language = None):
    #     non_prt = re.compile(r'[\W]+')
        if language in TrainingFeature.NLTK_LANGS:
            tknLang = TrainingFeature.NLTK_LANGS[language]
        else:
            tknLang = TrainingFeature.NLTK_LANGS['??']

        en_stpwrds = stopwords.words('english')
        nl_stpwrds = stopwords.words('dutch')

    #     print('detect: ',detect(inp_text))
    #     print('langs detect: ',detect_langs(inp_text))
    
        text = inp_text.lower()
        
        # tokenize input
        tknd_list1 = word_tokenize(text, language=tknLang)
        
        regex = re.compile('[%s]' % re.escape(string.punctuation+string.digits)) #see documentation here: http://docs.python.org/2/library/string.html
        
        tknd_list2 = []
        for token in tknd_list1:
            new_token = regex.sub(u'', token)
            if not new_token == u'' and len(token) > 1:
                tknd_list2.append(new_token)
    
        tknd_list3  = [x for x in tknd_list2 if x not in nl_stpwrds]            
        stpd_list   = [x for x in tknd_list3 if x not in en_stpwrds]
        
    #     if language == 'dutch':
    #         txt_stemmer = DutchStemmer()
    #     else:
    #         txt_stemmer = EnglishStemmer()
    #     stmd_list = [txt_stemmer.stem(tkn) for tkn in tknd_list2]
    
        stpd_text = ' '.join(stpd_list)
   
        return stpd_list, stpd_text

        
#--------------------------------------------------------------------------
#--     
#--------------------------------------------------------------------------  
def build_htmltext_feature(pageFeatures, featureCollection, write_to_db = False):
  
    if  ( len(pageFeatures.html_text) > 5 ):
        trnFeature = TrainingFeature('HtmlText',pageFeatures)
        trnFeature.as_htmltext(pageFeatures)
        featureCollection.addTrainingInstance(trnFeature)
#         if write_to_db:
#             trnfeaturesTbl[url] = trnFeature        
        return True
    else:
        print('         !!!!! site has no html text !!!!!',len(pageFeatures.html_text))
        return None

#--------------------------------------------------------------------------
#--     
#--------------------------------------------------------------------------        
def build_metadata_feature(pageFeatures, featureCollection, write_to_db = False):


    trnFeature = TrainingFeature('MetaData',pageFeatures)
    trnFeature.as_metadata(pageFeatures)
    featureCollection.addTrainingInstance(trnFeature)
#         if write_to_db:
#             trnfeaturesTbl[url] = trnFeature        
#         return True
    return


#--------------------------------------------------------------------------
#--     
#--------------------------------------------------------------------------
def build_metatext_feature(pageFeatures, featureCollection, write_to_db = False):
    if (len(pageFeatures.meta_text) > 5) or (len(pageFeatures.html_title)> 5):
        
        trnFeature = TrainingFeature('MetaText' , pageFeatures)
        trnFeature.as_metatext(pageFeatures)
          
        if len(trnFeature.data) <= 5:
            print('         !!!!! Result Data is <= 5 bytes and will be skipped')
            pass
        else:
            featureCollection.addTrainingInstance(trnFeature)
#             if write_to_db:
#                 trnfeaturesTbl[url] = trnFeature    
            return True
    else:
#         print('   $$$$$ site has no MetaKwrds/title text $$$$$',len(pageFeatures.meta_text))
        pass
    # end-if
    return False


#--------------------------------------------------------------------------
#--   append a training Feature to it's Feature Set collection 
#-------------------------------------------------------------------------- 

    def save_to_db(self):
        trnFeaturesTbl = MongoTrnFeatures()
        trnFeaturesTbl[self.url] = self.to_json()
        return 
    
    
    def __repr__(self):
        return "<Item {} with URL {} Page Title {}>".format(self._id, self.url, self.html_title)

        

#--------------------------------------------------------------------------
#  get <link> realated information 
#--------------------------------------------------------------------------
           
    def to_json(self):
#         self.get_page_metadata_info()
        return {
            "url"           :   self.url,
            "label"         :   self.label,
            "label_cd"      :   self.label_cd,
            "feature_name"  :   self.feature_name,
            "data_lang"     :   self.data_lang,
            "data"          :   self.data,
            'timestamp'     :   datetime.now()            
        }
    
#--------------------------------------------------------------------------
#--     
#--------------------------------------------------------------------------  
    @classmethod
    def get_by_id(cls, _id):
        item_data = Database.find_one("pagefeatures", {"_id": _id})
        return cls(**item_data)
      
