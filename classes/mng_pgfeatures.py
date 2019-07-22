'''
Created on Apr 4, 2017

@author: kevin.bardool
'''

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


class MongoPageFeatures(object):
    """
    Wrapper around MongoDB to PageFeatures collection

    """
# possible states of a webpage
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
        self.db.webpages.create_index([('url', pymongo.ASCENDING)],unique=True)
        self.timeout = timeout
        return
 
    def __contains__(self, url):
        try:
            self[url]
        except KeyError:
            return False
        else:
            return True
    
    def __getitem__(self, _id):
        """Load value at this URL
        """
        print('----- lookup :',_id)
        record = self.db.pagefeatures.find_one({'_id': _id})
        
        if record:
            return record
        else:
            raise KeyError(_id+ ' does not exist')


    def __setitem__(self, _id, record):
        """Save value for this URL
        """
#         p = result if len(result)< 10 else len(result)
#         print("      Pagefeatures _setitem_ with id:", _id, 'features: ',features )
        
        self.db.pagefeatures.update({'_id': _id}, {'$set': record}, upsert=True)

    def openCursor(self,cond=None,proj=None):
        Cursor = self.db.pagefeatures.find(filter=cond,projection=proj)
        return Cursor

    def getNextFromCursor(self,Cursor):
        record = Cursor.next()
        if (record):
            return record
        else:
            return None
         
    def count(self,condition):
        rc = self.db.pagefeatures.count(condition)
        return rc
    
#     def parsed(self, url):
#         self.db.crawl_queue.update({'_id': url}, {'$set': {'status': self.PARSED}})
        