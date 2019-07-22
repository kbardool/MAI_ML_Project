# import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class MongoQueue(object):
    """
        Wrapper around MongoDB to crawl queue collection
    """
    # possible states of a download
    OUTSTANDING = 0
    PROCESSING  = 1
    COMPLETE    = 2
    FAILED      = 3 
    
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
        record = self.db.crawlqueue.find_one(
            {'status': {'$ne': self.COMPLETE}, 'training':1} 
        )
        return True if record else False

    def __getitem__(self, _id):
        """Load value at this URL
        """
        record = self.db.crawlqueue.find_one({'_id': _id})

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
                  'dnld_ts'   : None                  
                 }  
        
        self.db.crawlqueue.update({'_id': _id}, {'$set': record}, upsert=True)


    def openCursor(self,cond=None,proj=None):
        Cursor = self.db.crawlqueue.find(filter=cond,projection=proj)
        return Cursor

    def push(self, _id, content):
        """Add new URL to queue if does not exist
        """
        try:
            self.db.crawlqueue.insert(
                {'_id'       : _id, 
                 'training'  : content['training'],
                 'label'     : content['label'],
                 'status'    : self.OUTSTANDING, 
                 'timestamp' : datetime.now(),
                 'dnld_ts'   : None
                 })
#             print(' Row with id:',_id, ' successfully added')
            return True
        except  DuplicateKeyError as e:
#             print(' Row with id:',_id, ' already exists...will be skipped')
            return False

    def pop(self, training_mode):
        """Get an outstanding URL from the queue and set its status to processing.
           If the queue is empty a KeyError exception is raised.
        """
        training_val = 1 if training_mode else 0
        
        record = self.db.crawlqueue.find_one_and_update(
            {
              'status':  self.OUTSTANDING,'training': training_val
            }, 
            {'$set': 
              {'status': self.PROCESSING, 'timestamp': datetime.now()}
            }
        )
        
        if record:
            return record
        else:
            raise KeyError()
        
        
        
    def remove(self,_id):
        """Get an outstanding URL from the queue and set its status to processing.
           If the queue is empty a KeyError exception is raised.
        """
        rc = self.db.crawlqueue.delete_one({'_id': _id})
        
#         print(' remove return code is ', rc.deleted_count)
        if rc.deleted_count == 1:
            return rc.deleted_count
        else:
            raise KeyError()
        
        
    def peek(self,training_mode):
        training_val = 1 if training_mode else 0
        record = self.db.crawlqueue.find_one({'status': self.OUTSTANDING, 'training':training_val})
        if record:
            return record['_id']
        
    def count_Outstanding(self, training_mode):
        training_val = 1 if training_mode else 0
        return self.db.crawlqueue.count({'status': self.OUTSTANDING,'training':training_val})
                   
                
    def count(self):
#         record = self.db.crawlqueue.find_one({'status': self.OUTSTANDING})
#         if record:
        return self.db.crawlqueue.count()
    


    def outstanding(self, _id):
        self.db.crawlqueue.update({'_id': _id}, 
                                    {'$set': 
                                      {'status': self.OUTSTANDING,  'dnld_ts':datetime.now()}
                                    })
    
    def failure(self, _id, rc=None):
        self.db.crawlqueue.update({'_id': _id}, 
                                    {'$set': 
                                      {'status': self.FAILED,
                                       'dnld_ts':datetime.now(),
                                       'rc':rc}
                                    })

    
    def complete(self, _id, rc=None):
        self.db.crawlqueue.update({'_id': _id},
                                    {'$set': 
                                      {'status': self.COMPLETE,
                                       'dnld_ts':datetime.now(),
                                       'rc': rc}
                                    })
 
            
    def repairAll(self):
        """Release stalled jobs
        """
        print('     ------------------------------------------------------------------------------')
        print('     Repair URLs in outstanding or failed status in crawler queue ')
        print('     Repair URLs with timestmap prior to: ', datetime.now() - timedelta(seconds=self.timeout))

        # record1 = self.db.crawlqueue.update_many(
            # {
                # 'timestamp': {'$lt': datetime.now() - timedelta(seconds=self.timeout)},
                # 'status': {'$eq': self.PROCESSING}
            # },
            # {
                # '$set': {'status': self.OUTSTANDING, 'dnld_ts':None}
            # })
            # print('     URLs in Processing status reset: %i'%(record.modified_count))            
            
        record = self.db.crawlqueue.update_many(
            {
                'timestamp': {'$lt': datetime.now() - timedelta(seconds=self.timeout)},
                'status': {'$in': [self.PROCESSING, self.FAILED]}
            },
            {
                '$set': {'status': self.OUTSTANDING, 'dnld_ts':None}
            })
        
        if record:
            print('     URLs Reset : %i'%( record.modified_count))
            print('     Repair URLs in Outstanding/Failed Status Complete ')        

        else:
            print('     Repair URLs in Outstanding/Failed Status Failed ')        
        print('     ------------------------------------------------------------------------------')            
        return
        
        
    def repairAllFailed(self):
        """Reset failed downloads jobs
        """
        print('------------------------------------------------------------------------------')
        print('  Repair URLs in FAILED status in crawler queue ')
#         print('  self timeout is     : ',self.timeout)
#         print('  delta selftimeout is: ',timedelta(seconds=self.timeout))
        print('  Repair URLs with timestmap prior to: ',
              datetime.now() - timedelta(seconds=self.timeout))

        record = self.db.crawlqueue.update_many(
            {
                'timestamp': {'$lt': datetime.now() - timedelta(seconds=self.timeout)},
                'status': {'$eq': self.FAILED}
            },
            {
                '$set': {'status': self.OUTSTANDING, 'dnld_ts':None}
            })
        
        if record:
            print('  Records searched: %i Records Modified: %i'%(record.matched_count, record.modified_count))
            print('  Repair URLs in Failed Status Complete ')
            print('------------------------------------------------------------------------------')
    

    def clearTable(self):
        self.db.crawlqueue.drop()