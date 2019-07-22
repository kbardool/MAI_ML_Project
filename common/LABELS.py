'''
Created on Apr 12, 2017

@author: kevin.bardool
'''
LABEL_NAMES      = [
    'company',               
    'error'  ,                         
    'for sale',               
    'holding page company',   
    'holding page non-commercial',  
    'non-commercial' ,        
    'password protected',     
    'pay-per-click',          
    'personal-family-blog',   
    'portal/media' ,          
    'web-shop' ]


LABEL_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


LABEL_NAME_TO_CD      = {
    'company'               : 0,
    'error'                 : 1,         
    'for sale'              : 2,
    'holding page company'  : 3,
    'holding page non-commercial' : 4,
    'non-commercial'        : 5,
    'password protected'    : 6,
    'pay-per-click'         : 7,
    'personal-family-blog'  : 8,
    'portal/media'          : 9,
    'web-shop'              : 10    }

LABEL_CD_TO_NAME = {
    0 :'company',               
    1 :'error'  ,                         
    2 :'for sale',               
    3 :'holding page company',   
    4 :'holding page non-commercial',  
    5 :'non-commercial' ,        
    6 :'password protected',     
    7 :'pay-per-click',          
    8 :'personal-family-blog',   
    9 :'portal/media' ,          
    10:'web-shop'               
    }

BINARY_LABEL_NAMES = ['non-company','company']        

TEXT_LANGUAGES     = ['nl', 'en', 'fr', 'de', 'da', 'af']

TAG_LIST = [
    'article', 'aside', 'b',  'br',    'button',    'div',    'form', 'footer',     
    'img',    'input',    'li', 'link',  'meta',    'nav',    'script',     
    'style',  'header',   'h1', 'h2', 'h3', 'h4', 'h5', 'i','iframes', 'option', 'p',
    'pswd_fld', 'section', 'select',    'small',    'strong',    'span',
    'table',     'ul'    ] 

HTTP_ERRORS = {'forbidden'            : 'Forbidden', 
              'access forbidden'     : 'Forbidden',
              'host not found'       : 'Host not found',
              'service unavailable'  : 'Service Unavailable',
              'service temporarily unavailable' : 'Service Unavailable',
              'canceled'             : 'Refused',
              'closed'               : 'Refused',
              'refused'              : 'Refused',
              'connection timed out' : 'Connection Timeout',
              'timeout'              : 'Connection Timeout',
              'internal server error': 'Internal Server Error', 
              'not found'            : 'Not Found' ,
              '404 not found'        : 'Not Found' ,
              'no response received' : 'No Response Received',
              'internal server error': 'Internal Server Error'  }     
