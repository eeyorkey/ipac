import os, sys

class DataParserWeight:
    def __init__(self):
        self.name = "data_loader_weight"
        
    def load_data(self, data_file):
        print self.name
        
        if os.path.isfile(data_file) == False:
            sys.stderr.write("the file %s does not exist" % data_file)
            sys.exit(1)
            
        f_reader = open(data_file)
        data = []
        for each_line in f_reader:
            data.append(each_line.strip())
            
        f_reader.close()
        return data