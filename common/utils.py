#-------------------------------------------------------------------------------
# Util functions
#-------------------------------------------------------------------------------
import re, sys, locale,os
import urllib, zlib, bson
from   sys      import stdout
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
def unzip_html(zipped_html):
    html_dict    = bson.BSON.decode(zlib.decompress(zipped_html))
    return html_dict['html']

def write_bytestream(filepath=None,filenm=None,content=None):
    with open(filepath+filenm,'wb' ) as f_obj:
        f_obj.write(content)
        f_obj.close()    
    return

def write_strstream(filepath=None,filenm=None, content=None, encoding='utf-8'):
    with open(filepath+filenm, 'w' , encoding=encoding) as f_obj:
        f_obj.write(content)
        f_obj.close()    
    return

def write_picklefile(filepath=None,filenm=None,content=None):
    with open(filepath+filenm+'.pkl','wb') as f_obj:  
        pickle.dump(content, f_obj, pickle.HIGHEST_PROTOCOL)
        f_obj.close()    
    return

def write_stdout(filepath=None,filenm=None,stdout=stdout):
    with open(filepath+filenm+'.out','wb'  ) as f_obj :
        content = stdout.getvalue().encode('utf_8')
        f_obj.write(content)
        f_obj.close()    
    return

def write_zip_stdout(filepath=None,filenm=None,stdout=stdout):
    with open(filepath+filenm+'_xtrct_out.zip','wb'  ) as f_obj :
        stdout = stdout.getvalue().encode('utf_8')
        comp_stdout = zlib.compress(stdout)   
        f_obj.write(comp_stdout)
        f_obj.close()    
    return


def normalize(seed_url, link):
    """Normalize this URL by removing hash and adding domain
    """
    link, _ = urllib.parse.urldefrag(link) # remove hash to avoid duplicates
    return urllib.parse.urljoin(seed_url, link)


def same_domain(url1, url2):
    """Return True if both URL's belong to same domain
    """
    return urllib.parse.urlparse(url1).netloc == urllib.parse.urlparse(url2).netloc


def get_robots(url):
    """Initialize robots parser for this domain
    """
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
    rp.read()
    return rp
        

def get_links(html):
    """Return a list of links from html 
    """
    # a regular expression to extract all links from the webpage
    webpage_regex = re.compile('<a[^>]+href=["\'](.*?)["\']', re.IGNORECASE)
    # list of all links from the webpage
    print('in getlineks ',type(html))
    return webpage_regex.findall(str(html))


def print_encoding_info(): 
    print(' Stdout encoding is : ........... ',sys.stdout.encoding)
    print(' System default encoding is : ... ',sys.getdefaultencoding())
    print(' Preferred encoding is : ........ ',locale.getpreferredencoding())
    print(' Default locale: ................ ',locale.getdefaultlocale())
    print(' PYTHONIOENCODING:............... ',os.environ["PYTHONIOENCODING"]) 
    print('------------------------------------------------------------------')
    return 

def strip_ctrls(input):

    if input:

        import re

        # unicode invalid characters
        RE_XML_ILLEGAL = u'([\u0000-\u0008\u000a-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                         u'|' + \
                         u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                          ( chr(0xd800), chr(0xdbff), 
                            chr(0xdc00), chr(0xdfff),
                            chr(0xd800), chr(0xdbff), 
                            chr(0xdc00), chr(0xdfff),
                            chr(0xd800), chr(0xdbff), 
                            chr(0xdc00), chr(0xdfff),
                           )
        input = re.sub(RE_XML_ILLEGAL, " ", input)

        # ascii control characters
#         input = re.sub(r"[\x01-\x1F\x7F]", "", input)

    return input

'''
Class recieves a filename. filename is in CSV format

callback module :
    opens file ,
        extracts URLs
        if not seen URL :
            loads in mongo queue
            adds to url_list
        else
            prints message and skips
    return url_list
    
@author: kevin.bardool
@date:   Apr 7, 2017

'''
import os, csv
# from zipfile import ZipFile, 
# import StringIO, BytesIO
from classes.mng_queue      import MongoQueue
from classes.mng_domains    import MongoDomains
from pymongo.errors         import DuplicateKeyError
from common.LABELS                 import LABEL_NAMES,LABEL_NAME_TO_CD

 

#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------        
def load_TrainingData(input_file, header = None, read_limit = 100000, write_limit= 10000, output = None):
    queueTbl      = MongoQueue()
    domainInfoTbl = MongoDomains()
    url_list = []       

    num_labels   = len(LABEL_NAMES)
    read_counts    = [0] * num_labels
    picked_counts  = [0] * num_labels
    write_counts   = [0] * num_labels
    nowrite_counts = [0] * num_labels
    read_sum       = 0
    write_sum      = 0
    nowrite_sum    = 0
    pcked_sum      = 0
    ttl_read_count = 0
    ttl_write_count= 0
    f_obj = open(input_file,newline='') 
    if output is not None:
        output_file = open(output, 'w' ,newline ='', encoding='utf-8')
        output_csv = csv.writer(output_file ,quoting=csv.QUOTE_MINIMAL)
    lines   = csv.reader(f_obj)
    
    #       skip header line if necessary         
    if header is not None:
        line = next(lines, None)            # skip header line
        print('      Skipping header line:',line)            
        
    line = next(lines, None)
    label_count  = {}        
    domain       = line[0].strip()
    tld          = line[1].strip()

    
    while line is not None:
        if (ttl_read_count == read_limit):
            print('     Hit max read limit of ', read_limit, 'rows')
            break
        ttl_read_count += 1
        if  domain == line[0].strip() and tld == line[1].strip():
            lbl_nm = line[2].strip()
            lbl_cd = LABEL_NAME_TO_CD[line[2].strip()]
            read_counts[lbl_cd] += 1
            # gather stuff about the website
            label_count[lbl_cd] = label_count[lbl_cd] + 1 if lbl_cd in label_count else 1
            line = next(lines, None)
            
        else:   
            url = domain+'.'+tld

            if len(label_count) == 1:
                sel_lbl_cd,_ = label_count.popitem()   
                sel_lbl_nm = LABEL_NAMES[sel_lbl_cd]
                picked_counts[sel_lbl_cd] +=1                    
            else:                    
                #decide on proper label
                print('  ',url,'    labels found :', label_count  )
                        
                max_cnt = 99999999
                for key,val in label_count.items():
                    print( ' key:',key , 'count:',val, ' samples picked from this key',picked_counts[key])
                    if picked_counts[key] < max_cnt:
                        max_cnt = picked_counts[key]
                        sel_lbl_cd = key
                # end for
                
#                     label_list = sorted(label_count.items(), key=lambda labelx: labelx[1], reverse=True )                      
#                     print('     sorted label_list is :', label_list )          
#                     i = iter(label_list)
#                     sel_lbl_cd = next(i)[0]  
                
                sel_lbl_nm = LABEL_NAMES[sel_lbl_cd]
                print('           selected label :', sel_lbl_nm,'-',sel_lbl_cd)                
                print('          new read_counts :', read_counts )                
                print('          old pckd_counts :', picked_counts )  
                picked_counts[sel_lbl_cd] +=1
                print('          new pckd_counts :', picked_counts )   
                
#               persist to queue table                
            try:
                # domainInfoTbl[url] = {'label' : sel_lbl_nm, 'training' : 1}  
                write_counts[sel_lbl_cd] += 1 
                url_list.append(url)
                ttl_write_count  += 1 
                if output is not None:
                    output_csv.writerow([url])                    
                if  (write_counts == write_limit):
                    print('     Hit max write limit of ',write_limit)
                    break                    
            except DuplicateKeyError  :
                nowrite_counts[sel_lbl_cd] += 1
                print('    <',url, '> is already in queue')
                print(' it will not be added to queue table  ')
            
#                 print(' new url encountered :',line)
            label_count = {}
            domain  = line[0].strip()
            tld     = line[1].strip()

        #end_if
    # end while      
    f_obj.close()
    if output is not None:
        output_file.close()
    
    
    totals = {'read':read_counts, 'picked':picked_counts, 'write':write_counts,
              'no_write': nowrite_counts}
    print('     Num of urls queued for processing: ',len(url_list) ,'\n')
    print('     -CD---------- LABEL --------------     ---Read---  ---Slctd---  --pct--   ---Wrttn---  ---Skipd---')
    num_labels   = len(LABEL_NAMES)
    for i in range(num_labels):

        print('      %2i %30s     %9i    %9i    %6.2f    %9i    %9i'%(i,
                        LABEL_NAMES[i],
                        totals['read'][i],
                        totals['picked'][i], 
                        totals['picked'][i]*100/totals['read'][i] if totals['read'][i] != 0 else 0,
                        totals['write'][i],
                        totals['no_write'][i] ))
                               
        read_sum    += totals['read'][i]
        write_sum   += totals['write'][i]
        nowrite_sum += totals['no_write'][i]
        pcked_sum   += totals['picked'][i] 
    print('     -------------------------------------------------------------------------------------------------')
    print('          %30s    %9i    %9i    %6.2f    %9i    %9i'%(
          'Totals:',read_sum, pcked_sum, pcked_sum * 100/ read_sum if read_sum != 0 else 0,  
                   write_sum, nowrite_sum))
    print('     -------------------------------------------------------------------------------------------------')    
    return url_list
    
#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------       
def load_TestData(input_file, header= None, read_limit=100000, write_limit=100000, output=None):
#         print("    In load testdata .. Reading from :", input_file)

    queueTbl      = MongoQueue()
    url_list       = []       
    read_counts    = 0
    write_counts   = 0 
    dup_counts     = 0  
    if output is not None:
        output_file = open(output, 'w' ,newline ='', encoding='utf-8')
        output_csv = csv.writer(output_file ,quoting=csv.QUOTE_MINIMAL)
    f_obj = open(input_file,newline='') 
    lines   = csv.reader(f_obj)

    #       skip header line if necessary         
    if header is not None:
        line = next(lines, None)            # skip header line
        print('     Skipping header line: ',line )
                
    line = next(lines, None)
    
    while line is not None:
        if (read_counts == read_limit):
            print('     Hit max read limit of ', read_limit, 'rows')
            break
                
        read_counts += 1
        url = line[0].strip() + '.' + line[1].strip()
        
        if  queueTbl.push(url, {'label' : None, 'training' : 0}):  
            write_counts  += 1 
            url_list.append(url)
            if output is not None:
                    output_csv.writerow([url])
            if  (write_counts == write_limit):
                print('     Hit max write limit of ',write_limit)
                break
        else:
            dup_counts  += 1
            print('     <',url, '> is already in queue - will be skipped')

        line = next(lines, None)
        #end_if
    # end while
    f_obj.close()
    if output is not None:
        output_file.close()
        

    print('     Num of urls queued for processing: ',len(url_list) ,'\n')
    print('     ----------------------------------     ---Read---   ---Dups---  ---Wrttn---')
    print('    %30s     %9i    %9i    %9i'%(
          ' Input urls read/wrtiten/skipped',read_counts, write_counts, dup_counts))
    print('    ----------------------------------------------------------------------------')
            
    return url_list 
                    