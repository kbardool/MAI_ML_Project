'''
Created on Mar 12, 2017

@author: kevin.bardool
'''
import json, time
# from classes.throttle               import Throttle
from datetime                       import datetime
from urllib.parse                   import urlparse, urlsplit
from selenium                       import webdriver
from selenium.webdriver.support.ui  import WebDriverWait
from common.utils                   import write_picklefile
from urllib.error                   import URLError

OUTPUT_PATH   = '../test_output/'    
DEFAULT_AGENT = 'wswp'
DEFAULT_DELAY = 5
DEFAULT_RETRIES = 1
DEFAULT_TIMEOUT = 30

#-------------------------------------------------------------------------------
# CLASS Downloader
#-------------------------------------------------------------------------------
class Downloader:
    def __init__(self, delay=DEFAULT_DELAY,  webdriver = None, proxy=None ,
                 num_retries=DEFAULT_RETRIES, timeout=DEFAULT_TIMEOUT, cache=None):

        self.webdriver   = webdriver
        self.throttle    = Throttle(delay)
        self.proxy       = proxy
        self.num_retries = num_retries
        self.cache       = cache
        print("    Downloader _init_ complete")
        return
    
    def __call__(self, url ):
#         print("        Downloader() Callback function for : ",url)
#         print('        Module name:', __name__,'parent process:', os.getppid(),'process id:', os.getpid())
        result = None

        if self.cache:
            try:
                result = self.cache[url]
                print('        Webpage on webPages table - last downloaded :',
                               result['timestamp'],'  Dnld RC: ',result['code'])
                result = None
            except KeyError:
                # url not found in webpages table 
                pass
                    
#           result was not loaded from webpages table so still need to download    
    
        if result is None:
            self.throttle.wait(url)
            result = self.dnld(url , num_retries=self.num_retries)
#           print('        Downloader() complete - code:',rc,' html size is ', len(self.webdriver.page_source))
        return result

    #-------------------------------------------------------------------------------
    # method - download
    #-------------------------------------------------------------------------------
    def dnld(self, url, num_retries, data=None):
        print('        Dnld() : Downloading:', url, 'try #: ',num_retries)
        
        code = None
        self.har = ''
        
        WebDriverWait(self.webdriver, 15)
        req_url  = 'http://www.'+ url 
        try:
            self.webdriver.get(req_url)
        except URLError as e:
            code = 999
            statusText = 'Undefined Error Occurred'
        else:
            har_json = self.webdriver.get_log('har')
            self.har = json.loads(har_json[0]['message'])

            if len(self.har['log']['entries']) != 0:
                code     = self.har['log']['entries'][0]['response']['status']
                statusText=self.har['log']['entries'][0]['response']['statusText']
                ret_url  = self.har['log']['entries'][0]['request']['url']
                print('        >>>> Download complete. Code:',code,'   StatusText:',statusText,'  status is: ', code)
    #           print('             requested url:',req_url , '      reply url:',ret_url)        
    #           print('            ',self.har['log']['entries'][0]['request'])
    #           print('            ',self.har['log']['entries'][0]['response'])
            else:
                code = 999
                statusText = 'No Response Received'
                ret_url = ''
                print('        !!!! No valid har returned  Code set to 999 Status to No Repsonse Received' )
                # write_picklefile(OUTPUT_PATH, url+'_har' ,self.har)
                # print('       wrote pickled output to:',url+'_har.pkl')            
        # end try - except - else
            
        if (code is None):
            code = 599
            
        if ( 199 < code < 400):
            html_content = self.webdriver.page_source
            if (req_url == ret_url.lower()):
                ret_url = ''
            else:    
                o1 = urlparse(req_url)
                o2 = urlparse(ret_url.lower())
                
                if (o1.netloc != o2.netloc):
                    print('             parse1',o1)
                    print('             parse2',o2)
                    code = 300
                    ret_url = o2.netloc
                    ret_url = ret_url.replace('www.','',1) if ret_url.startswith('www.') else ret_url
                else:
                    ret_url = ''  
        
        else :   # (code < 200 or code > 399):
            html_content = ''            
            print('        !!!! HAR Status code:  ', code, '    Status Text:', statusText,
                    '  Ret_url:', ret_url)
            
            if ( 400 <= code < 600  or code == 0) and  (num_retries > 0):
                # retry 5XX HTTP errors
                print('        !!!! Retry Download....', num_retries-1,' more times')
                return self.dnld(url, num_retries-1, data)
            # endif    
        # endif
        
        

        return ({'url': url, 'code':code, 'statusText':statusText, 
                 'redir_url': ret_url, 'html': html_content} )
        
        
#-------------------------------------------------------------------------------
# Class Throttle
#-------------------------------------------------------------------------------
class Throttle:
    '''
    classdocs
    '''

    """Throttle downloading by sleeping between requests to same domain
    """
    def __init__(self, delay):
        # amount of delay between downloads for each domain
        self.delay = delay
        # timestamp of when a domain was last accessed
        self.domains = {}
        
    def wait(self, url):        
        """Delay if have accessed this domain recently
        """
#         print('        ---- Throttle Delay for : ',url)
        domain =  urlsplit(url).netloc        
        last_accessed = self.domains.get(domain)

        if self.delay > 0 and last_accessed is not None:
            sleep_secs = self.delay - (datetime.now() - last_accessed).seconds
            if sleep_secs > 0:
                time.sleep(sleep_secs)
        self.domains[domain] = datetime.now()
