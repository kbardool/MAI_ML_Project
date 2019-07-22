'''
Created on Apr 16, 2017

@author: kevin.bardool
'''
import os,  datetime,time
import multiprocessing
import threading

#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------
class ThreadProcess(threading.Thread):
    '''
    classdocs
    '''
    def __init__(self, name, started, queue, variable):
        """ This class represents a single instance of a running thread"""
        threading.Thread.__init__(self)
        self.setName(name) 
        self.name = name
        self.start_time = started
        self.queue = queue
        self.variable = variable
        print('      @@c Creating ',self.getName(),'@ ',self.start_time, 
              '  queue.len: ',queue.qsize(), ' variable:', variable)
        
        
        
    def run(self ):
        proc_ctr = 0
        print('      @@r ',self.name,'-',self.getName(), 'started at : ', self.start_time,
                      'Length of taskqueue is :', self.queue.qsize())
        
        while not self.queue.empty() :
            if proc_ctr > 10:
                print('\t\t',self.name,'-',self.getName(), ' processed 10 items from queue - stop thread.')
                break
            
            try:
                item = self.queue.get(True,3)
            except self.queue.Empty:
                    print('\t\t',self.name,'-',self.getName(),' Queue is empty, we will terminate the thread..')
                    break
            else:
                print('\t\t',self.name,'-',self.getName(),'  Var is :',self.variable)
                self.variable = self.variable + 1
                print('\t\t',self.name,'-',self.getName(),'  Process queue item -----------------',self.getName(),' ',self.variable)
                for k, i in item.items():
                    print('\t\t',self.name,'-',self.getName(),' Que key:',k , ' vlaues: ',i)
            time.sleep(10)
        #end while
        return    
            
            
#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------            
class ThreadDispatcher(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        print('* Thread Dispatcher ',self,' Created')
        
        
    def start(self, pid, numThreads, taskqueue):
        print('  ** Thread Dispatcher - setup threads for process:',pid)
#         print('     Length of taskqueue is :', taskqueue.qsize())
        myThreads = []
        
        for tid in range(numThreads):
            name = '$Proc-'+str(pid)+'-Thread-'+str(tid)
            start_time = datetime.datetime.now().strftime("%m%d%Y@%H%M")
            th = ThreadProcess(name, start_time,taskqueue,pid*10000)
            myThreads.append(th)
            
        print('  ** Thread Dispatcher - start threads for process:',pid)            
        for th in myThreads:
            print('    ** Start Thread  ',th)
            th.start()
            
        for t in myThreads:
            print('  ** call thread joiner for ',t.getName())
            t.join()
            
            
#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------
            
class ProcessDispatcher(object):
    ''' Create and start processes '''
    
    def __init__(self, numProcesses, numThreads, taskqueue):
        '''
        Constructor
        '''
        #    pool = multiprocessing.Pool(processes=num_cpus)
        print('* Number of processes (nun_cpus): ',numProcesses)

        processes = []
        trunner   = ThreadDispatcher()
    
        for pid in range(numProcesses):
            p = multiprocessing.Process(target=trunner.start, args=(pid+1,numThreads,taskqueue))
            processes.append(p)
            print("* process # ",pid+1,'created -  parent pId:', os.getppid(),'pId:', os.getpid())

        for p in processes:
            print('* Start Process ', p)
            p.start()
    
            # wait for processes to complete
        for p in processes:
            p.join()    