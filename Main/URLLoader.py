'''
Created on Mar 12, 2017

@author: kevin.bardool
'''
# -*- coding: utf-8 -*-
import zlib, bson, time, sys
# print('train Test_Loader __name__ is ',__name__)  
if __name__ == '__main__':
    REL_PATH = '../'
    sys.path.append(REL_PATH)
else:
    REL_PATH   = ''
    
# from crawler import crawler
import  argparse
import  sys, time
from datetime            import datetime
from classes.dataloader  import DataLoader 
from common.LABELS       import LABEL_NAMES
OUTPUT_PATH  = REL_PATH
INPUT_PATH   = REL_PATH
#--------------------------------------------------------------------------
#
#--------------------------------------------------------------------------
def URLLoader( input_file, training_mode = False, skip_header=True, read_limit=10000, write_limit=10000):
    
    print('\n --> URLLoader Started Process at: ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"))
    
    input_filename  = INPUT_PATH +  input_file
    output_filename = OUTPUT_PATH + 'crawlfile.csv'
    print()
    print('     Training Mode : ', training_mode)
    print('     Input file to be processed: ', input_file)
    print('     Skip header line:',' Yes ' if skip_header else 'No')
    print('     Read limit: ',read_limit,' Write limit: ',write_limit)
    start = time.time()

    loader = DataLoader(input_filename)
    if training_mode:
        urls = loader.load_TrainingData(header     = skip_header, 
                                        read_limit = read_limit,
                                        write_limit= write_limit,
                                        output     = output_filename)
    else :
        urls = loader.load_TestData(header     = skip_header, 
                                    read_limit = read_limit,
                                    write_limit= write_limit,
                                    output     = output_filename)

    elapsed_time = time.time() - start
    print('     Elapsed time: %.2f seconds'%(elapsed_time))
    print('\n --> URLLoader completed successfully at: ' ,
          datetime.now().strftime("%m-%d-%Y @ %H:%M:%S"),'\n')
    return urls
    
    
#--------------------------------------------------------------------------
#  Main Driver 
#--------------------------------------------------------------------------
if __name__ == '__main__':
    # Get address from command line.
    parser = argparse.ArgumentParser(description='Load csv file of websites to crawl')
    parser.add_argument('input_file', metavar='Filename', help='relative path in csv file')
    parser.add_argument('--train')
    args = parser.parse_args()
    args.train = True if args.train is not None else False
    print('   args parsed input parm is :' ,args.input_file,  ' train is : ',args.train)        
    
    URLLoader(args.input_file, training_mode = args.train, skip_header=True, read_limit=10, write_limit=10)
    exit(' URLLoader completed successfully ')
