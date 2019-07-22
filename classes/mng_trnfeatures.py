'''
Created on Apr 4, 2017

@author: kevin.bardool
'''

try:
    import cPickle as pickle
except ImportError:
    import pickle
# import  os, 
# import zlib,bson
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
 

class MongoTrnFeatures(object):
    """
    Wrapper around MongoDB to Training Features collection

    """
    # possible states of a feature set
    NOT_PROCESSED  = 0 
    PROCESSING    = 1
    PROCESSED     = 2


    def __init__(self, client=None, timeout=300):
        """
        host: the host to connect to MongoDB
        port: the port to connect to MongoDB
        timeout: the number of seconds to allow for a timeout
        """
        self.client = MongoClient(host='localhost', port=27017) if client is None else client
        self.db = self.client.cache
        self.db.trnfeatures.create_index([('url', pymongo.ASCENDING),
                                          ('feature_name', pymongo.ASCENDING)],
                                          unique=True)
        self.timeout = timeout
        return 
 
#     def __contains__(self, url):
#         try:
#             self[url]
#         except KeyError:
#             return False
#         else:
#             return True
    
    def __getitem__(self, _id):
        """Load value at this URL
        """
        print('----- lookup :',_id)
        record = self.db.trnfeatures.find_one({'_id': _id})
         
        if record:
            return record
        else:
            raise KeyError(_id+ ' does not exist')


    def __setitem__(self, _id, features):
        """Save value for this URL
        """
#         print("      trnfeatures. _setitem_ with id:", url, 'features: ',features )
        record = {
            "label"         :   features['label'],
            "data"          :   features['data'],
            "status"        :   self.NOT_PROCESSED,
            'timestamp'     :   datetime.now()
        }
        
        rc = self.db.trnfeatures.update_one(
                                {'url': url, 'feature_name': features['feature_name']} ,
                                {'$set': record}, 
                                upsert=True)
#         print('rc 2:',rc.modified_count)
        return 


    def insert(self,features):
        """Save value for this URL
        """
#         print("      trnfeatures. _setitem_ with id:", url, 'features: ',features )

        record = {
            "url"           :   features['url'],
            "feature_name"  :   features['feature_name'],        
            "label"         :   features['label'],
            "data"          :   features['data'],
            "status"        :   self.NOT_PROCESSED,
            'timestamp'     :   datetime.now()
        }
 
        rc = self.db.trnfeatures.insert_one(record)
#         print('rc 2:',rc.inserted_id)
        return 
    
    
    def get_by_feature_name(self,feature_name):
        Cursor = self.db.trnfeatures.find({'feature_name':feature_name})
        return Cursor
    
    
    def get_by_label(self,label):
        Cursor = self.db.trnfeatures.find({'label':label})
        return Cursor
    
    
    def openCursor(self,condition):
        Cursor = self.db.trnfeatures.find(condition)
        return Cursor


    def getNextFromCursor(self,Cursor):
        record = Cursor.next()
        if (record):
            return record
        else:
            return None
         
         
    def count(self,condition):
        rc = self.db.trnfeatures.count(condition)
        return rc
    
#     def parsed(self, url):
#         self.db.crawl_queue.update({'_id': url}, {'$set': {'status': self.PARSED}})
        