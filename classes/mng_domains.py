# import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class MongoDomains(object):
    """
    Wrapper around MongoDB to training Domain Names collection
    """
    # possible states of a download
    OUTSTANDING     = 0
    ADDED_TO_QUEUE  = 1
    PROCESSED_OK    = 2
    PROCESSED_FAILED= 3

    
    def __init__(self, client=None, timeout=300):
        """
        host: the host to connect to MongoDB
        port: the port to connect to MongoDB
        timeout: the number of seconds to allow for a timeout
        """
        self.client = MongoClient(host='localhost', port=27017) if client is None else client
        self.db = self.client.cache
        self.timeout = timeout
        return

    def __nonzero__(self):
        """Returns True if there are more jobs to process
        """
        record = self.db.domains.find_one(
            {'status': {'$ne': self.COMPLETE}, 'training':1} 
        )
        return True if record else False


    def __getitem__(self, _id):
        """Load value at this URL
        """
        record = self.db.domains.find_one({'_id': _id})

        if record:
            return record
        else:
            print("Queue _getitem_ with Id:",_id,'   not found')
            raise KeyError(_id + ' does not exist')
        
        
    def __setitem__(self, _id, content):
        """Save value for this URL
        """
#         print("      Webpages _setitem_ with url:",url, 'result', p )
        record = {
                  'status'    : self.OUTSTANDING,
                  'training'  : content['training'],
                  'label'     : content['label'],
                  'timestamp' : datetime.now(),
                  'dnld_ts'   : None,
                  'dnld_rc'   : None  
                 }  
        
        return self.db.domains.update({'_id': _id}, {'$set': record}, upsert=True)

    def openCursor(self,cond=None,proj=None):
        Cursor = self.db.domains.find(filter=cond,projection=proj)
        return Cursor

    def count(self):
#         record = self.db.domains.find_one({'status': self.OUTSTANDING})
#         if record:
        return self.db.domains.count()

    
    def count_Outstanding(self):
        return self.db.domains.count(
                {'status': self.OUTSTANDING,'training':1})

            
    def peek(self):
        record = self.db.domains.find_one(
                                   {'status': self.OUTSTANDING, 'training':1}
                                )
        if record:
            return record['_id']
        
        
    def outstanding(self, _id):
        self.db.domains.update({'_id': _id}, 
                                    {'$set': 
                                      {'status': self.OUTSTANDING,
                                       'dnld_ts':datetime.now()}
                                    })
    
    
    def in_queue(self, _id):
        self.db.domains.update({'_id': _id}, 
                                    {'$set': 
                                      {'status': self.ADDED_TO_QUEUE,
                                        'dnld_ts':datetime.now()}
                                    })
    
    
    def complete(self, _id, rc=None):
        self.db.domains.update({'_id': _id},
                                    {'$set': 
                                      {'status': self.PROCESSED_OK,
                                       'dnld_ts':datetime.now(),
                                       'rc': rc}
                                    })
 
    def failed(self, _id, rc=None):
        self.db.domains.update({'_id': _id},
                                    {'$set': 
                                      {'status': self.PROCESSED_FAILED,
                                       'dnld_ts':datetime.now(),
                                       'rc': rc}
                                    })
 
  
 
    
    def clear(self):
        self.db.domains.drop()
        