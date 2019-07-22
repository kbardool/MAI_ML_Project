import os
# import csv
# from zipfile import ZipFile
# from io import StringIO,BytesIO


class SiteScraper:
    def __init__(self, max_urls=10):
        self.max_urls = max_urls
        # print("    Scrape Agent (SiteScraper) _init_ complete")
        return
        
    '''
    Parse Results from html download
    '''
    def __call__(self, url, result):
        html = result['html']
        code = result['code']
        
        print("        @@@@ In scrape agent(SiteScraper) callback .. URL:",url,
              'HTML length:',len(html), ' HTML Code:',code)
#         print('        module name:', __name__,'parent process:', os.getppid(),'process id:', os.getpid())
#         urls = []       
        TtlAdded = 0
        
#         cache = MongoCache()

#         StrIO_html = BytesIO(html)

#         with ZipFile(StrIO_html) as zf:
#             with zf.open(zf.namelist()[0]) as csv_file:
#                 for line in csv_file.readlines() :
#                     p = line.decode().splitlines()[0].split(',')
#                     website = p[1]
 
#                     if 'http://' + website not in cache:
#                         urls.append('http://' + website)
#                         print('    +++++ Adding: ',website, 'to URL list')
#                         TtlAdded += 1
#                         if len(urls) == self.max_urls:
#                             print('    hit max url limit of ',self.max_urls)
#                             break
#                     else:
#                         print('    <',website, '> is already in cache ')

        print("        @@@@ Scrape Agent(SiteScraper) callback complete :",TtlAdded)
        return 
    