'''

Read HTML from webPages table
    Extract various feature sets (eg. HTML Text, Meta info,.....)
    write to pageFeatures table
    
Created on Mar 7, 2017

@author: kevin.bardool
'''
import zlib, bson, time, sys
# print('train DataExtraction __name__ is ',__name__)  
if __name__ == '__main__':
    REL_PATH = '../'
    sys.path.append(REL_PATH)
else:
    REL_PATH   = './'
    

from classes.mng_domains     import MongoDomains
from classes.mng_webpages    import MongoWebpages
from classes.mng_trnfeatures import MongoTrnFeatures
from classes.page_features   import PageFeatures
from classes.training_set    import TrainingSet
from classes.training_feature import build_htmltext_feature, build_metatext_feature, build_metadata_feature
from common.LABELS           import LABEL_NAMES,LABEL_NAME_TO_CD
from datetime                import datetime


DEFAULT_PROCESS_COUNT = 10000
TRAINING_FS_PATH  = 'output/Training/'
TEST_FS_PATH      = 'output/Test/'
DBG_FILENAME = 'dbg_file'
CSV          = '.csv'   
'''
#--------------------------------------------------------------------------
#  Main Routine 
#--------------------------------------------------------------------------'''    
def DataExtraction(training_mode = False, write_to_db = False, process_limit = DEFAULT_PROCESS_COUNT, Urllist= None):
    
    print('\n --> Data Extraction1 Started at:', datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    if training_mode:
        output_path = REL_PATH + TRAINING_FS_PATH
    else:
        output_path = REL_PATH + TEST_FS_PATH
        
    file_pfx = 'TRNFS_' if training_mode else 'TSTFS_'
    file_sfx = datetime.now().strftime("%m%d@%H%M")
    dbg_filename = output_path + file_pfx + DBG_FILENAME + file_sfx+CSV
    dbg_file = open(dbg_filename, 'w' ,newline ='', encoding='utf-8')
            
    print('     Training Mode is    :  ', training_mode)
    print('     Write to DB is      :  ', write_to_db)
    print('     Process count limit :  ', process_limit)
    print('     Output files path   :  ', output_path)
    print('     Debug file name is  :  ', dbg_filename)
    
    start = time.time()
            
    sitelistTbl = MongoDomains()
    webpagesTbl = MongoWebpages()
    trnfeaturesTbl  = MongoTrnFeatures()
        
    webpages_read       = 0
    html_short          = 0
    html_goodcode       = 0 
    html_bad_code       = 0 
    html_processed      = 0

    processed           = 0
    not_processed       = 0
    siteinfo_notfound   = 0    
    training            = 0
    non_training        = 0
    html_text           = 0
    no_html_text        = 0
    tag_counts          = 0 
    no_tag_counts       = 0 
    meta_text           = 0 
    no_meta_text        = 0 
    count_by_label      = [ 0 for _ in LABEL_NAMES]
    MetaText   = TrainingSet()
    HtmlText   = TrainingSet()
    MetaData   = TrainingSet()
#     cond = {'training': 1,'code':{'$gt':300}}  #  'label':"error", 
#     cond = {'status': {'$ne': 2}, 'training' : 1}


    if training_mode:
        cond = {'training': 1}
    else :   
        cond = {'_id': {'$in':Urllist}}
    

    MyCursor = webpagesTbl.openCursor(cond)
    cursorCount = MyCursor.count()
    if MyCursor is None :
       exit('Failed to open Cache cursor')
    print('     Number of cursor rows returned: ', cursorCount)

    for i,webpage in enumerate(MyCursor):
        
        if (webpage is None) or  (i == process_limit):
            break    
        
        webpages_read += 1

        print('     Webpage    : ', i, '      _id : ' , webpage['_id'])
        url = webpage['url']    
        
        if training_mode:
            try:
                siteinfo = sitelistTbl[url]
            except KeyError:
                print('     Siteinfo not found on Domain Table for:', url)
                siteinfo_notfound += 1
                continue
            else:
                if  (siteinfo['training']==0):
                    print('     Not a training url -- skip') 
                    non_training += 1
                    continue
                else:
                    label = siteinfo['label']
                    label_cd = LABEL_NAME_TO_CD[label]
                    count_by_label[label_cd] +=1
                    training += 1
        else :
            non_training +=1      
            try:
                siteinfo = sitelistTbl[url]
                label = siteinfo['label']
                label_cd = LABEL_NAME_TO_CD[label]
            except KeyError:
                label_cd = '------'
                pass
        #end-if
        # print('     Webpage    : ', i, '      _id : ' , webpage['_id'], 'Label :',label_cd,'-',label)
        html_dict    = bson.BSON.decode(zlib.decompress(webpage['html']))
        webpage['html'] = html_dict['html']
        html_len     = len(webpage['html'])
        
#         print('   Trn: ', 'YES' if siteinfo['training'] == 1 else 'no' , '  label :', label)        
#         print('   St_Cd : ', webpage['code'],'    St_Txt : ', webpage['statusText'],
#               '  HTML len : ',html_len, '  URL : ', webpage['url'],'  Redir URL : ',webpage['redir_url'])
#         print('   Downloaded : ', webpage['timestamp'], '  Webpage row status : ', webpage['status'])
                

        if not ( 200<= webpage['code']<=300):
            html_bad_code += 1
        elif (html_len < 30):
            html_short+= 1    
        else:
            html_goodcode +=1
        
        # extract_features from downloaded webpage
        pageFeatures = PageFeatures(webpage, debug=False, dbgfile = dbg_file)
        
        if training_mode and write_to_db:
            pageFeatures.save_to_db()
            webpagesTbl.parsed(url)
 
        # build feature sets from webpage extracted data
        if build_htmltext_feature(pageFeatures, HtmlText) :
            html_text += 1
        else:
            no_html_text +=1
        
        if build_metatext_feature(pageFeatures, MetaText) :
            meta_text +=1
        else:
            no_meta_text +=1
        
        if build_metadata_feature(pageFeatures, MetaData) :
            tag_counts += 1
        else:
            no_tag_counts +=1

        html_processed +=1
    # end for    
     
    HtmlText.write_to_CSV(file_pfx+'HtmlText', output_path, file_sfx)
    HtmlText.write_to_Pkl(file_pfx+'HtmlText', output_path, file_sfx)
 
    MetaText.write_to_CSV(file_pfx+'MetaText' , output_path, file_sfx)
    MetaText.write_to_Pkl(file_pfx+'MetaText' , output_path, file_sfx)

    MetaData.write_to_CSV(file_pfx+'MetaData', output_path, file_sfx)    
    MetaData.write_to_Pkl(file_pfx+'MetaData', output_path, file_sfx)

    FeatureFiles = {'HtmlText': output_path+file_pfx+'HtmlText'+file_sfx+'.pkl',
                    'MetaText': output_path+file_pfx+'MetaText'+file_sfx+'.pkl',
                    'MetaData': output_path+file_pfx+'MetaData'+file_sfx+'.pkl'}
    
    print('\n\n')
    print('     ----- Data Extraction Results -------------------------------')
    print('       Result webpages from cursor:................. ', cursorCount)
    print('       Webpages read :.............................. ', webpages_read)
    if training_mode:    
        print('       Webpages not found on queue :................ ', siteinfo_notfound )
        print('       Webpages with training = 0 :................. ', non_training)
    #end-if
    print('     - Input Details totals --------------------------------------')
    print('       HtmlPage return  w/ html < 30 bytes :........ ', html_short)
    print('       Bad html status code :....................... ', html_bad_code)
    print('       Good html status code ( 200~300):............ ', html_goodcode)
    print('       Input Records processed :.................... ', html_processed)   
    print('       Total Domains Read :......................... ', training)
    print('     -------------------------------------------------------------') 
    if training_mode:
        print('\n')
        print('     ------------- LABEL --------------        --- Count ---')
        for i in range(len(count_by_label)):
            print('     %2i %30s  \t\t %5i'%(i,LABEL_NAMES[i],count_by_label[i]))    
    #end-if
 
    print('     ----- Feature Generation Results ----------------------------')
    print('       siteinfo not found: ......................... ', siteinfo_notfound)    
    print('       training rows: .............................. ', training)    
    print('       non training rows: .......................... ', non_training)    
    print('       pageFeatures processed:...................... ', processed)   
    print('       rows with html text: ........................ ', html_text) 
    print('       rows without html text: ..................... ', no_html_text) 
    print('       rows with tag counts: ....................... ', tag_counts) 
    print('       rows without tag counts: .................... ', no_tag_counts) 
    print('       rows with meta keywords/title: .............. ', meta_text) 
    print('       rows without meta_keywords/title: ........... ', no_meta_text) 
    print('       pageFeatures rows NOT processed: ............ ', not_processed)
    print('       Cursor processed for: ....................... ', processed+not_processed, ' rows')           
    print('     -------------------------------------------------------------\n')
    for i in FeatureFiles:
        print('     ',i,'feature file written to : ', FeatureFiles[i])

    elapsed_time = time.time() - start
    
    print('     Elapsed time: %.2f seconds'%(elapsed_time))
    print('\n --> Data Extraction1 ended at:',datetime.now().strftime("%m-%d-%Y @ %H%M%S"))
    print()
    dbg_file.close()
    return FeatureFiles

#--------------------------------------------------------------------------
# driver for direct call
#--------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--db')
    parser.add_argument('--limit')
    args = parser.parse_args()

    print(args.train)
    print(args.db)
    print(args.limit)
    print(sys.argv)
    
    tm = args.train  if args.train is not None else False
    pl = args.limit  if args.limit is not None else DEFAULT_PROCESS_COUNT
    db = args.db     if args.db    is not None else False
    DataExtraction(training_mode=tm, write_to_db = db, process_limit = pl)

    exit(' Data Extraction Program completed successfully')