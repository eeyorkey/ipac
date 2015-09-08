import sys, os
from data_loaders import *

class DataLoaderScheduler:
    def __init__(self):
        '''
        register data parsers
        '''
        self.pval_parser = data_loader_pval.DataParserPval()
        self.weight_parser = data_loader_weight.DataParserWeight()
        self.annotation_parser = data_loader_annotation.DataParserAnnotation()
        self.parsers = {\
            "data_loader_pval":self.pval_parser,\
            "data_loader_weight":self.weight_parser,\
            "data_loader_annotation":self.annotation_parser\
            }
        
    def get_parser(self):
        if self.parsers.has_key(self.data_parser):
            return self.parsers[self.data_parser]
        else:
            sys.stderr.write("Unkown parser:%s\n" % parser)
            sys.exit(1)
            
    def get_data(self, f_data, parser):
        self.f_data = f_data
        self.data_parser = parser
        
        if os.path.isfile(f_data) == False:
            sys.stderr.write("either data or parser does not exist")
            sys.exit(1)
            
        parser = self.get_parser()
        return parser.load_data(self.f_data)
