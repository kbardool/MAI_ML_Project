'''
Created on Apr 8, 2017

@author: kevin.bardool
'''
 
try:
    import cPickle as pickle
except ImportError:
    import pickle

import zlib, bson
from datetime import datetime
import pymongo
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


class MongoWebpages(object):
    """
    Wrapper around MongoDB to cache downloads
 
    """
    # possible states of a webpage
    NOTPROCESSED  = 0 
    PROCESSING    = 1
    PARSED        = 2
    

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
    
    
    def __getitem__(self, url):
        """Load value at this URL
        """
        record = self.db.webpages.find_one({'url': url})
 
        if record:
            return record
        else:
#             print("         Webpages _getitem_ with url:",url,'   not found')
            raise KeyError(url + ' does not exist')


    def __setitem__(self, url, result):
        """Save value for this URL
        """
        encd_html = bson.BSON.encode({'html':result['html']})
        comp_html = zlib.compress(encd_html)
        
        record = {'timestamp' : datetime.now(), 
                  'url'       : result['url'],
                  'status'    : self.NOTPROCESSED, 
                  'statusText': result['statusText'],
                  'code'      : result['code'], 
                  'redir_url' : result['redir_url'],
                  'training'  : result['training'],
                  'label'     : result['label'],
                  'html'      : comp_html}
        
        self.db.webpages.update({'_id': url}, {'$set': record}, upsert=True)

    def update(self, url, cond):
        rc = self.db.webpages.update({'_id': url}, {'$set': cond})
        return rc
    
    def update_label(self, url, label):
        """Save value for this URL
        """
        record = {'label' : label }
        rc = self.db.webpages.update({'_id': url}, {'$set': record})
        return rc
    
    def openCursor(self,cond=None,proj=None):
        Cursor = self.db.webpages.find(filter=cond,projection=proj)
        return Cursor

    def getNextFromCursor(self,Cursor):
        try:
            record = Cursor.next()
            html_dict = bson.BSON.decode(zlib.decompress(record['html']))
            record['html'] = html_dict['html']
            return record
        except StopIteration:
            print(' hit the end of the cursor')
            return None
         
    def count(self,condition):
        rc = self.db.webpages.count(condition)
        return rc
    
    def parsed(self, url):
        rc = self.db.webpages.update({'_id': url}, {'$set': {'status': self.PARSED}})
        return rc
        