'''
Created on Mar 13, 2017

@author: kevin.bardool
'''
# -*- coding: utf-8 -*-
import multiprocessing
import os, json
import sys
import threading
import time

from selenium                       import webdriver
from selenium.webdriver.support.ui  import WebDriverWait

from classes.downloader             import Downloader 
from classes.mng_domains import MongoDomains
from classes.mng_queue              import MongoQueue
from classes.mng_webpages           import MongoWebpages
from classes.site_scraper           import SiteScraper


try:
    import cPickle as pickle
except ImportError:
    import pickle
    
SLEEP_TIME      = 10
DEFAULT_DELAY   = 5
DEFAULT_RETRIES = 1
DEFAULT_TIMEOUT = 10
DEFAULT_THREADS = 1
                
'''
#------------------------------------------------------------------------------
# Thread process - extract url from queue and download
#------------------------------------------------------------------------------'''            
def process_queue(Tname, training ,crawlqueueTbl, webpagesTbl, siteinfoTbl, downloadAgent, scraperAgent):
        
    print("    ===> Create New Process Queue Thread ",Tname,'/',os.getpid())
    print('         Training Mode is ',training )
    processed   = 0
    successful  = 0
    unsuccessful = 0
    
    while True:
      
        try:
            crawl_item = crawlqueueTbl.pop(training)
            url = crawl_item['_id']
            processed += 1
        except KeyError:        # currently no urls to process
            break
        else:
            print('    ++++ download:',url )
            result = downloadAgent(url)  
            download_rc  = result['code']
            result['training'] = 1 if training else 0
            result['label']    = crawl_item['label']
            print('    ++++ RC : ', download_rc ,
                  '    Resp Len: ', len(result['html']), '  ',url)
    
        #-----------------------------------------------------------------            
        #  save result to cache            
        #-----------------------------------------------------------------
        if webpagesTbl:
    #                 print('        ---- save result to webpagesTbl')
            webpagesTbl[url] = result
              
        if (200 <= download_rc <= 300) :
            if (scraperAgent):
                try:
                    links = scraperAgent(url, result) or []
                except Exception as e:
                    print('     @@@@ Error in Scraper callback for: ',url,'Exception ', e)
                    continue
            # end-if
    
            # mark this URL as crawled
            if training :
                try:              
                    siteinfoTbl.complete(url,download_rc)
                    # print(' $$$$$$$$$$ mark ',url ,' on siteinfo table as complete  ')
                except Exception as e:
                    print('     @@@@ Error in updating SiteInfo tbl: ',url,'Exception ', e)
            #end if
             
            try:
                crawlqueueTbl.remove(url)
            except Exception as e:
                print('     @@@@ Error in updating crawl queue tbl: ',url,'Exception ', e)
            else:
                successful +=1
#                         # add the new links returned from the URLscraper to the queue
#                         for link in links:
#                             # add this new link to queue
#                             crawlqueueTbl.push(normalize(url, link),
#                                                {'label' : None, 'training' : 0})
        else:
            print('    Error in download process - Scraper callback skipped :', download_rc)
            crawlqueueTbl.failure(url,download_rc)
            unsuccessful +=1       
        # end if     
    # end while .True.
        
    print('     ---' , Tname,' crawling results --------------')
    print('      processed   : ' , processed)
    print('      successful  : ' , successful)
    print('      unsuccessful: ' , unsuccessful)
    print('     ----', Tname, ' ending normally ---------------')
    return
                
'''
#------------------------------------------------------------------------------
# Threaded Crawler Manager: Initialize and dispatch threads
#------------------------------------------------------------------------------'''              
def threadManager( Pname ,training = False,   
#                  cache = None,   scraperAgent = None, 
                   delay      = DEFAULT_DELAY ,   num_retries= DEFAULT_RETRIES, 
                   max_threads= DEFAULT_THREADS,  timeout    = DEFAULT_TIMEOUT ):
    """
    Build Webdriver for this process , then initiate multiple Crawl threads
    """
    print('*** ', Pname,'(',os.getpid(),') Thread Manager Process ----------')
    print('    Training mode is ', training)
 
    scraperAgent  = SiteScraper()
    webpagesTbl   = MongoWebpages()
    siteinfoTbl   = MongoDomains()
    crawlqueueTbl = MongoQueue(timeout=12)
    wbdrvr = webdriver.PhantomJS(service_args = ['--ignore-ssl-errors=yes' ])    
    wbdrvr.set_window_size(1400, 1050)
    WebDriverWait(wbdrvr, 15)    
    
    downloadAgent =  Downloader(cache        = webpagesTbl, 
                                webdriver   = wbdrvr, 
                                delay       = delay,
                                num_retries = num_retries, 
                                timeout     = timeout)
        
    #----------------------------------------------------------------
    # wait for all download threads to finish
    #----------------------------------------------------------------
    threads = []
    queue_count = crawlqueueTbl.count_Outstanding(training)

    
    while threads or (queue_count > 0):

        print('   *** PId: ', Pname, '(',os.getpid(),')  polling threads & crawlqueue :' ,
                 queue_count, ' outstanding rows in crawlqueueTbl')
        
        for thread in threads:
            if not thread.is_alive():
                print("    ===> remove thread")
                threads.remove(thread)
                
        print('   *** PId: ', Pname, '(',os.getpid(),') Max Threads:',max_threads,'  current num of threads', len(threads))  
              
        while len(threads) < max_threads and crawlqueueTbl.peek(training):
            # can start some more threads
#             print("    ===> create new thread for process queue ")
            Tname = Pname + ' T-'+ str(len(threads))
            thread = threading.Thread(target=process_queue, 
                     args = (Tname, training, crawlqueueTbl, webpagesTbl, siteinfoTbl, downloadAgent, scraperAgent))
            thread.setDaemon(True) # set daemon so main thread can exit when receives ctrl-c
            thread.start()
            threads.append(thread)
            
        time.sleep(SLEEP_TIME)
        queue_count = crawlqueueTbl.count_Outstanding(training)
    #end while
    print('   *** PId: ', Pname, '(',os.getpid(),') No more threads or queue is empty')
    wbdrvr.quit()
    return


'''
#------------------------------------------------------------------------------
# Process_Crawler: Initialize and dispatch processes
#------------------------------------------------------------------------------'''
def crawl_ProcessManager(*args, **kwargs):
    print('     >>  Process Manager Input args are        :',args)
    print('     >>  Process Manager Input keyword args are:',kwargs)
    
    num_processes = kwargs.pop('num_processes',0)
    
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
        print('     >  Num_processes not specified - will be set to cpu_count: ',num_processes)
    else:
        print('     >  Number of processes set to: ',num_processes)

    processes = []

    for i in range(num_processes):
        Pname = 'P-'+str(i+1)
        p = multiprocessing.Process(target=threadManager, name = Pname, args = (Pname,), kwargs =  kwargs )
        #parsed = pool.apply_async(threaded_link_crawler, args, kwargs)
        processes.append(p)
        print("     >>   Process ",p.name, 'initialized.  parent pId:', os.getppid(),'pId:', os.getpid())

    for p in processes:
        print('     >>   Start process : ',p)
        p.start()
    
    # wait for processes to complete
    for p in processes:
        p.join()

    return
        
        
        
#---------------------------------------------------------------------------------- 
#                 pklname = "..\\output\\"+url+"_har.pickle"
#                 with open(pklname, 'wb') as f:
#                     pickle.dump(result['har'], f, pickle.HIGHEST_PROTOCOL)
#                     f.close()
#                 
#                 jsonname = "..\\output\\"+url+"_har.json"
#                 har_data = json.dumps(result['code'], indent=4)
#                 with open(jsonname, 'wb') as f:
#                     f.write(har_data)
#                     f.close()
#                                     
#                 save_har = open(harname, "a")
#                 print(' wrote 2')
#                 htmlname = "..\\output\\"+url+".html"
#                 with open(htmlname, 'wb') as f:
#                     f.write(result['html'].encode('utf-8'))
#                     f.close()
#----------------------------------------------------------------------------------