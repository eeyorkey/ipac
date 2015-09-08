import os, sys

class DataParserAnnotation:
    def __init__(self):
        self.name = "data_loader_annotation"
        
    def load_data(self, data_file):
        print self.name
        
        if os.path.isfile(data_file) == False:
            sys.stderr.write("the file %s does not exist" % data_file)
            sys.exit(1)
            
        f_reader = open(data_file)
        # skip the first line, which is the header
        line_id = 0
        data = []
        for each_line in f_reader:
            line_id += 1
            if line_id == 1:
                continue
            
            s_list = each_line.strip().split(" ")
            content = []
            for i in range(0, len(s_list)):
                content.append(s_list[i])
                
            data.append(content)
        
        f_reader.close()
        
        return data