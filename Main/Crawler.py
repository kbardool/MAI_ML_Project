'''
Download webpages for URLs save in crawler_queue
Created on Mar 13, 2017

@author: kevin.bardool
'''
# -*- coding: utf-8 -*-

import sys,time
# print('train Crawler __name__ is ',__name__)  
if __name__ == '__main__':
    REL_PATH = '../'
    sys.path.append(REL_PATH)
else:
    REL_PATH   = './'
    
from datetime               import datetime
from Main.CrawlProcess      import crawl_ProcessManager
from classes.mng_queue      import MongoQueue
# from classes.site_scraper import SiteScraper

#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------    
def CrawlUrls (training=False, max_processes = 2, max_threads=1):
    print(' --> Crawler Process Started : ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    start = time.time()
    
    print('     Training mode is : ', training)
    print('     Max Processes    : ', max_processes,' Max Threads:',max_threads)
    crawl_queue = MongoQueue(timeout=12)
        
    print('     Before Repairs:  There are %i outstanding row(s) in queue'%
                crawl_queue.count_Outstanding(training) )

    crawl_queue.repairAll()
 
    
    print('     After Repairs:  There are %i outstanding row(s) in queue'%
                crawl_queue.count_Outstanding(training) ) 

    crawl_ProcessManager(training=training, num_processes=max_processes , 
                         max_threads=max_threads , timeout=99)  

    elapsed_time = time.time() - start
    print('     Elapsed time: %.2f seconds'%(elapsed_time))
    
    print('\n --> Crawler Process Ended : ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    return True

#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--procs')
    parser.add_argument('--threads')
    args = parser.parse_args()

    print(args.train)
    print(args.procs)
    print(args.threads)
    print(sys.argv)
    tm = args.train   if args.train is not None else False
    mp = args.procs   if args.procs is not None else 2
    mt = args.threads if args.threads is not None else 1
 

    CrawlUrls( training=tm, max_processes= mp, max_threads=mt)

    exit(' Crawler1 completed successfully')
