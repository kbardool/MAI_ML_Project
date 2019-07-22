'''
Created on Apr 6, 2017

@author: kevin.bardool
'''
__author__ = 'KBardool'

import io, bs4, bson, zlib , sys, re, urllib,unicodedata,math
import csv, string
import  common.utils as utils
from datetime import datetime   
from common.database            import Database
from classes.mng_pgfeatures     import MongoPageFeatures
from langdetect                 import detect
from common.LABELS               import LABEL_NAMES,LABEL_NAME_TO_CD 
from common.LABELS               import TEXT_LANGUAGES, TAG_LIST, HTTP_ERRORS 
from langdetect.detector_factory import detect_langs

# from models.stores.store import Store


    
#--------------------------------------------------------------------------
#  init routine
#--------------------------------------------------------------------------   
class PageFeatures(object):
    
    
    OUTPUT_PATH   = '../output/Debug/'
    NOT_PROCESSED = 0
    PROCESSED     = 1

    # Some strings for ctype-style character classification
    # whitespace = ' \t\n\r\v\f'
    # ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    # ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # ascii_letters   = ascii_lowercase + ascii_uppercase
    # digits          = '0123456789'
    # hexdigits       = digits + 'abcdef' + 'ABCDEF'
    # punctuation     = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # unicd_punct     = re.compile(r'[\u2018\u2019]')
    # ascii_cc        = re.compile(r'[\x01-\x1F\x7F]')    
    # printable       = digits + ascii_letters + punctuation + whitespace
    
    html_tags  = '<[a-zA-Z!\/][^>]*>' 
    wtspc_qt   = '[\s\"]+'
    non_alphanum = re.compile(r'[^%s]'% re.escape(string.digits + string.ascii_letters ))    
    punct_ws     = re.compile('[%s]+' % re.escape(string.punctuation))
    ws_qt        = re.compile('%s' % wtspc_qt)
    rem_www      = re.compile(r'www.')
    rem_url      = re.compile(r'^https?:\/\/.*[\r\n]*') 
    rem_dnlderr1  = re.compile(r'^Error downloading http\S+ - server replied: ')
    rem_dnlderr2  = re.compile(r'http\S+')
#     rem_dnlderr3  = re.compile(r'- server replied: ')     
    
    embded_html  = re.compile(html_tags)   
    unicd_cc     = re.compile(r'[\u2018\u2019\u0000-\u0008\u000a-\u000c\u000e-\u001f\ufffe-\uffff\ue000-\uf8ff]')
    
    
#--------------------------------------------------------------------------
#  init routine
#--------------------------------------------------------------------------    
    def __init__(self, webpage, debug = False, dbgfile=None, output_path = OUTPUT_PATH):
        
        self.debug          = debug
        if self.debug:
            sys.stdout = io.StringIO()

        self.dbgfile        = dbgfile
        self.output_path    = output_path
        self.csv_writer     = csv.writer(self.dbgfile ,quoting=csv.QUOTE_MINIMAL)

        self._id            = webpage['url']   #  uuid.uuid4().hex if _id is None else _id
        self.url            = webpage['url']
        self.wwwUrl         = self.url if self.url.startswith('www.') else 'www.'+self.url
        self.training       = webpage['training']
        
        if webpage['label'] is not None:
            self.label      = webpage['label']
            self.label_cd   = LABEL_NAME_TO_CD[self.label]
        else:
            self.label      = None 
            self.label_cd   = None
        
        self.html           = webpage['html']
        self.sp             = bs4.BeautifulSoup(self.html,'lxml' ) 
        
        self.tag_counts     = {}   
        self.meta_data      = {}    
        
        self.html_title     = ''
        self.html_text      = ''
        self.html_comments  = ''        
        self.meta_text      = ''
        self.frame_srcs     = []       
        self.anchor_hrefs   = []
        self.frame_hrefs    = []
        self.link_hrefs     = []
        self.anchor_texts   = []

        self.status         = PageFeatures.NOT_PROCESSED  
         
        # meta data fields --------------------------
        self.code           = webpage['code'] 
        self.statusText     = webpage['statusText'] 
        self.html_len       = len(self.html) if self.html is not None else 0         
        self.html_lang      = '??'
        self.text_lang      = '??'
        self.links_int      = 0
        self.links_ext      = 0     
        self.links_styl     = 0     
        self.ahref_int      = 0
        self.ahref_ext      = 0

        self.get_textual_info()
        self.get_html_lang()  
        if self.html_lang != self.text_lang:
            self.detect_content_lang()              
        self.get_html_title()
        self.get_metatag_info()
 
        self.get_frame_info()
        self.get_links_info()
        self.get_anchor_info()        
        self.get_tag_counts()
        self.get_page_metadata_info()
        
        if self.debug:
            self.disp_page_info()            
            self.write_html()         # <--- write complete html content to file
#             self.write_html_text()    # <--- write html text as pickle file


#    write CSV files to visualize features, if needed ..

        # if (len(self.html_text) > 0 ):
            # self.csv_writer.writerow([self.url,self.html_len,self.label,len(self.html_text) ,self.html_lang, self.text_lang, self.html_text])                    
        
#         if (len(self.html_comments) > 0 ):
#             self.csv_writer.writerow([self.url,self.html_len,self.label,len(self.html_comments) , self.html_comments])                    
        
#         if (len(self.meta_text) > 0 ):
#             self.csv_writer.writerow([self.url,self.html_len,self.label,len(self.meta_text) , self.meta_text])                    
        
#         if (len(self.html_title) > 0 ):
#             self.csv_writer.writerow([self.url,self.html_len,self.label,len(self.html_title) , self.html_title])                    

        self.csv_writer.writerow([self.url,self.label, self.code, self.statusText, self.text_lang,self.html_lang])
 


        # reroute stdout if we;re in debug mode                             
        if self.debug:
            utils.write_stdout(self.output_path, self.url, sys.stdout )        
            sys.stdout = sys.__stdout__
        
        return
    

#--------------------------------------------------------------------------
#  html classes
#--------------------------------------------------------------------------
    def get_page_metadata_info(self):

        self.meta_data['code']       = self.code
        
        if self.code == 200:
            self.statusText = 'OK'
        elif self.code == 300:
            self.statusText = 'Redirected'
        elif self.code == 599:
            self.statusText = self.rem_dnlderr1.sub('', self.statusText)
            self.statusText = self.rem_dnlderr2.sub('', self.statusText)
#             self.statusText = self.rem_dnlderr3.sub('', self.statusText)
            self.statusText = self.punct_ws.sub('', self.statusText).lower()
            if self.statusText in HTTP_ERRORS:
                self.statusText = HTTP_ERRORS[self.statusText]
            else:
                self.statusText = 'Other Errors'
        
        self.meta_data['statusText'] = self.statusText
        self.meta_data['html_len']   = self.html_len
        self.meta_data['text_lang']  = self.text_lang            
        self.meta_data['links_ext']  = self.links_ext
        self.meta_data['links_int']  = self.links_int
        self.meta_data['ahref_ext']  = self.ahref_ext
        self.meta_data['ahref_int']  = self.ahref_int
 
        self.meta_data['pswd_fld']   = len(self.sp.find_all("input",type="password"))
        for tag in TAG_LIST:
            self.meta_data[tag] = self.tag_counts.get(tag,0)
        
#         print(' Meta data info is: ', self.meta_data)  
          
        return  
    

#--------------------------------------------------------------------------
#  get textual info
#--------------------------------------------------------------------------
    def get_textual_info(self):
        
#         print('length of head: ',  'Content:', self.sp.head)
#         print('\n--- Collect Text      --------------------------------------')      

        text = io.StringIO()
        
        #-------------------------------------------------------------------------------------        
        # ignore and discard the comments for the time being as the do not provide useful info        
        #         comments = io.StringIO()
        #-------------------------------------------------------------------------------------
        html = self.sp.find_all("html")
    
        if len(html) > 0 :
            for idx, d in enumerate(html[0].descendants):
                desc_type   = type(d)
                parent_name = d.parent.name
                
                if self.debug and (desc_type is bs4.element.Tag) :
                    self.disp_tag_info(idx, d)
                
                elif desc_type is bs4.element.NavigableString:
                    str_len = len(d.string)
                    
                    if (parent_name in ['script', 'style', 'noscript','title']):
#                         print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>')                          
#                         print('   @@@@@@ NavigableString is child of <',parent_name,'> and is ignored')
                        continue
                    elif (str_len > 50000):
#                         print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>')                          
#                         print('   @@@@@@ Unusually long string',str_len,' > 50000 and  is ignored')
                        continue
                    #end_if

                    strng1 = self.embded_html.sub(' ', d.string)
                    strng2 = self.unicd_cc.sub(' ',strng1)                    
                    strng2 = self.punct_ws.sub(' ',strng2)
                    strng2 = self.ws_qt.sub(' ',strng2)
                    
                    if len(strng2) <= 2 :
#                         print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>')                        
#                         print('# ',idx, ' NavigableString length Empty or < 2 ignored ')
                        pass
                    else:            
#                         print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>')
#                         print('   NavigStr Bfr is: >',len(d.string),'< >',d.string,'<')      
#                         print('   NavigStr Aft is: >',len(strng2),'< >',strng2,'<')        
                        text.write(strng2.lower()+' ')
                    # end-if
                    
                    continue

                #-------------------------------------------------------------------------------------
                # ignore comment text for the time being as it does not provide useful information
                # elif desc_type is bs4.element.Comment:
                #     self.get_comment_text()
                #     continue
                #-------------------------------------------------------------------------------------
        
                # end if-elif
            # end for
        
        #end if len(html) > 0
        
#             self.html_text.replace('\xa0',' ')
#             self.html_text = '\"' + text.getvalue() + '\"'
        
        self.html_text =  text.getvalue()
        self.html_text = unicodedata.normalize("NFKD", self.html_text)
        text.close()
                                
#             self.html_comments = comments.getvalue().replace('\xa0',' ') 
#             comments.close
        if (self.html_text is not None and len(self.html_text)>10):
            self.text_lang = detect(self.html_text)
            if self.text_lang == 'af' :
                self.text_lang = 'nl'
        
        if self.html_text.find('\xa0') >= 0:
                print('         !!!!! \xa0 found !!!!!')
        return

#--------------------------------------------------------------------------
#  get html title information
#--------------------------------------------------------------------------
    def get_html_title(self):
        #--- Html Title -------------------------------------------
        if self.sp.title is not None:
            tg = self.sp.title.get_text() 
            tg = self.unicd_cc.sub(' ',tg)  
            tg = self.punct_ws.sub(' ',tg)
            tg = self.ws_qt.sub(' ', tg)
            self.html_title = tg.lower()         
        return
    
    
#--------------------------------------------------------------------------
#  get <link> realated information 
#--------------------------------------------------------------------------
    def get_html_lang(self):
        lang = '??'
        #--- Language Info ----------------------------------------
        if self.sp.html is not None:
            try:
                lang = self.sp.html['lang']
                lang = self.ws_qt.sub('', lang).lower()
            except KeyError:
                try:
                    lang = self.sp.html['xml:lang']
                    lang = self.ws_qt.sub('', lang).lower()
                except KeyError:
                    pass
            if len(lang) < 2:
                self.html_lang = '??'
            else:
                self.html_lang = lang[0:2]

        return
    
#--------------------------------------------------------------------------
#  Detect Content Language
#--------------------------------------------------------------------------    
    def detect_content_lang(self):
        self.text_lang = '??'
        
        if (self.html_text is not None and len(self.html_text)>10):
            langs  = detect_langs(self.html_text)
          
            for ln in langs:
                if ln.lang in TEXT_LANGUAGES:
                    self.text_lang = ln.lang
#                     prob = ln.prob
                    break
        #end-if
        
        if self.text_lang == 'af' :
            self.text_lang = 'nl'
            
        if self.html_lang == '??':
            if self.text_lang == '??':
                print('         !!!!! could not determine language (both are unknown) !!')
                print('         content lang: ',self.text_lang,'  len:',len(self.html_text),'  content:',self.html_text[0:100])
                print('         html_lang:', self.html_lang)                 
            else:
#                 print('   +++++ html_lang ?? will be set to :',self.text_lang)
#                 print('        detected text langs: ', langs) 
                self.html_lang = self.text_lang
        else:
            if self.text_lang == '??':
                print('         !!!!! html_lang :', self.html_lang, '   text_lang : ??')
                print('         set text_lang to match html_lang ')
                self.text_lang = self.html_lang
            else:
                print('         !!!!! html_lang :', self.html_lang, '   text_lang :',self.text_lang)
                print('         detected text langs: ', langs) 
                print('         Set html_lang to match text_lang')
#                 print('        content len:',len(self.html_text),'  content:',self.html_text[0:100])                
                self.html_lang = self.text_lang                 
        # end-if
        
        return
        
#--------------------------------------------------------------------------
#  get <link> realated information 
#--------------------------------------------------------------------------
    def get_links_info(self):
        self.link_style = len(self.sp.find_all("link", rel='stylesheet'))
        self.link_hrefs = [tt["href"] for tt in self.sp.find_all("link", href=True)]

        
        for link in self.link_hrefs:
            hostnm = urllib.parse.urlparse(link).hostname
            if hostnm is None:
                link_loc = None 
            else:
                link_loc = self.rem_www.sub('',hostnm)
                    
#               print('   link href   ',len(link), ' ' ,link)

            if (link_loc is None) or (link_loc == self.url) :
#                 print('   ','nothing' if  link_loc =='' else link_loc,'--',self.url,' INT LINK')     
                self.links_int += 1               
            else:
#                 print('   ','nothing' if link_loc  =='' else link_loc,'--',self.url,' EXT lINK')                    
                self.links_ext += 1                           
        return

    
#--------------------------------------------------------------------------
#  get frame information
#-------------------------------------------------------------------------- 
    def get_frame_info(self):
        self.frame_hrefs = [tt["src"] for tt in self.sp.find_all("frame", src=True)]        
    

#--------------------------------------------------------------------------
#  get anchor information
#--------------------------------------------------------------------------    
    def get_anchor_info(self):
        self.anchor_texts = [a.string   for a in self.sp.select("a") if a.string is not None]
        self.anchor_hrefs = [tt["href"] for tt in self.sp.find_all("a", href=True)]
        
        for link in self.anchor_hrefs:
            hostnm = urllib.parse.urlparse(link).hostname
            if hostnm is None:
                link_loc = None 
            else:
                link_loc = self.rem_www.sub('',hostnm)
                    
            if (link_loc is None) or (link_loc == self.url) :
                self.ahref_int += 1               
            else:
                self.ahref_ext += 1
                  
        return    


#--------------------------------------------------------------------------
#  get img information
#--------------------------------------------------------------------------    
    def get_image_info(self):
        self.image_hrefs = [tt["href"] for tt in self.sp.find_all("a", href=True)]
        

        for link in self.anchor_hrefs:
            hostnm = urllib.parse.urlparse(link).hostname
            if hostnm is None:
                link_loc = None 
            else:
                link_loc = self.rem_www.sub('',hostnm)
#                     print('   anchor href   ',len(link), ' ' ,link)
                    
            if (link_loc is None) or (link_loc == self.url) :
                self.ahref_int += 1               
            else:
                self.ahref_ext += 1
                  
        return        

    
#--------------------------------------------------------------------------
#  get meta tag information
#--------------------------------------------------------------------------         
    def get_metatag_info(self):
#         print('\n--- Collect meta tag info -------------------------------')

        def has_meta(tag):
            return tag.name == 'meta' and tag.has_attr('name') and  tag.has_attr('content')
        
        metas = self.sp.find_all(has_meta)
    
        for meta in metas:
            meta_name = meta["name"].replace('.',':').strip().lower()
            if meta_name in ['keywords', 'description']:
                
                meta_cntnt  = meta["content"].strip().lower()
#               print(meta["name"], '\t\t',len(meta_cntnt),'>>',meta_cntnt)
#               meta_cnt  =  unicodedata.normalize("NFKD", meta_cnt)
                meta_cntnt  = self.unicd_cc.sub(' ',meta_cntnt)
                meta_cntnt  = self.punct_ws.sub(' ',meta_cntnt)
                meta_cntnt  = self.ws_qt.sub(' ',meta_cntnt)                
                if len(meta_cntnt) > 1:
                    self.meta_text= self.meta_text  + meta_cntnt.lower() + ' '                
        return        
        

    
#--------------------------------------------------------------------------
#  get tag count information
#-------------------------------------------------------------------------- 
    def get_tag_counts(self):

        self.tag_counts = {}
        for i in  [x for x in self.sp.find_all(True)]:
            tg = self.non_alphanum.sub('',i.name)
            self.tag_counts[tg] = self.tag_counts[tg]+1 if tg in self.tag_counts else 1
            
        if self.debug :
            print('----- Tags count in document  -----------------------------------------------')             
            print(self.tag_counts)        


#--------------------------------------------------------------------------
#  convert to json
#-------------------------------------------------------------------------- 
    def to_json(self):
#         self.get_page_metadata_info()
        return {
#             "_id"           :   self._id,
            "url"           :   self.url,
            "training"      :   self.training,
            "label"         :   self.label,
            "label_cd"      :   self.label_cd,
            "code"          :   self.code,
            "statusText"    :   self.statusText,
            "html_title"    :   self.html_title,
            "html_lang"     :   self.html_lang,          
            "html_text"     :   self.html_text,
            "text_lang"     :   self.text_lang,
            "html_comments" :   self.html_comments,

            "frame_hrefs"   :   self.frame_hrefs,

            "anchor_hrefs"  :   self.anchor_hrefs,
            "anchor_texts"  :   self.anchor_texts,
            "meta_text"     :   self.meta_text,
            "tag_counts"    :   self.tag_counts,     
            "meta_data"     :   self.meta_data,       
            "status"        :   self.NOT_PROCESSED,
            'timestamp'     :   datetime.now()            
        }
    
    

#--------------------------------------------------------------------------
#  save to Mongo DB table
#--------------------------------------------------------------------------     
    def save_to_db(self):

        pagefeaturesTbl = MongoPageFeatures()
        pagefeaturesTbl[self.url] = self.to_json()
        return 
    
    
    def __repr__(self):
        return "<Item {} with URL {} Page Title {}>".format(self._id, self.url, self.html_title)


#--------------------------------------------------------------------------
#  display general info about extracted html information
#--------------------------------------------------------------------------   

    def disp_page_info(self):
        print("<Item {} with URL {} Page Title {}>".format(self._id, self.url, self.html_title))
        print('\n\n----- Results -----------------------------------------------------------')         
        print('Tag counts for html doc  : ', self.tag_counts)
        print('Html Language            : ', self.html_lang) # get title tag 
        print('Page Title               : ', self.html_title)
        print('Accumulated text         : ', self.html_text)
        print('Accumulated comments     : ', self.html_comments)
        print('Accumulated link hrefs   : ', self.link_hrefs)
        print('Accumulated frame hrefs  : ', self.frame_hrefs)
        print('Accumulated anchor hrefs : ', self.anchor_hrefs)
        print('Accumulated anchor texts : ', self.anchor_texts)
        print('Accumulated meta text    : ', self.meta_text)
        print('Accumulated meta data    : ', self.meta_data)
        print('       internal <links>  : ', self.links_int,
              '       external <links>  : ', self.links_ext)
        print('      internal <anchors> : ', self.ahref_int,
              '      external <anchors> : ', self.ahref_ext) 
        return 
    

#--------------------------------------------------------------------------
# write gathered text information from document to pickle file for debugging
#--------------------------------------------------------------------------   
    def write_html_text(self):
        utils.write_picklefile(self.output_path, self.url ,self.html_text)
#         print('   wrote pickled output to:',self.url+'.pickle')
 
    def write_html(self):
        utils.write_bytestream(self.output_path, self.url+'.html', self.sp.prettify("utf-8"))
#         print('   wrote prettified output to:',self.url+'.html')


    @classmethod
    def get_by_id(cls, _id):
        item_data = Database.find_one("pagefeatures", {"_id": _id})
        return cls(**item_data)


#--------------------------------------------------------------------------
# process tags
#--------------------------------------------------------------------------   

    def disp_tag_info(self,idx, d):                        
        num_childs = len(d.contents) 
        desc_type   = type(d)
        parent_name = d.parent.name
        
        if (d.name in ['script', 'noframes'] ) :
            print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>   #Children:',num_childs)                
            print('   @@@@@@ Script/noframes tag will be ignored ')
            pass
        elif d.name == 'meta':
            print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>   #Children:',num_childs)                
            print('   @@@@@@ Meta Tag    # Children: ' ,len(d.contents),'    Children: ' ,[x.name for x in d.contents])
            pass
        elif d.name == 'style':
            print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name,'>   #Children:',num_childs)                
            print('   @@@@@@ style Tag. Text is:',d.string)
            pass
        return 

#--------------------------------------------------------------------------
# process text found in comment tags
#--------------------------------------------------------------------------   
    
    def get_comment_text(self, idx, d):
        desc_type   = type(d)
        parent_name = d.parent.name
        
#       print('# ',idx, ' ',desc_type,'  ',' Name: <', d.name,'>  Parent: <',parent_name)
#       strng = d.string.replace('"',' ').replace('\n',' ').strip().lower()

        strng1 = self.embded_html.sub(' ', d.string)
        strng2 = self.unicd_cc.sub(' ',strng1)
        strng2 = self.ws_qt.sub(' ',strng2)
#         print('   comment is: >',len(strng2),'< >',strng2,'<')
         
        if (parent_name == 'script'):
#          print('# ',idx,'   Comment Child of script -ignored  Parent: <',parent_name,'>')
            pass
        elif (len(strng2) <= 1) :
#           print('# ',idx,'   Comment Empty - ignored    Parent: <',parent_name,'>')
            pass                
        else:
#           print('# ',idx,'   Comment Content: >',strg,'<   Parent: <',parent_name,'>')
#             comments.write(strng2+' ') 
            pass       
        return   
                        
                