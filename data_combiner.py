import os, sys

# combine pval, weight or annotation file and convert the file into the libSVM format
# supported by ipac c++
#
# if weight is empty, default weight is 1.0 for each instance
# if pval is empty, return error
# if annotation is empty, annotation is not used
#
#


class DataCombiner:
    def combine(self, data, data_out):
        if data.has_key("pval") == False:
            sys.stderr.write("pval information is missing\n")
            sys.exit(1)
        
        if data.has_key("weight") == False:
            sys.stderr.write("weight information is missing\n")
            sys.exit(1)
            
        if data.has_key("annotation") == False:
            sys.stderr.write("annotation information is missing\n")
            sys.exit(1)
            
        # begin to process
        weight_data = data["weight"]
        weight_len = len(weight_data)
        
        pval_data = data["pval"]
        pval_len_m = len(pval_data)
        if pval_len_m == 0:
            sys.stderr.write("pval data must exist\n")
            sys.exit(1)
        pval_len_n = len(pval_data[0])
        if pval_len_n == 0:
            sys.stderr.write("pval data must exist and cannot be empty\n")
            sys.exit(1)
            
        annotation_data = data["annotation"]
        annotation_len = len(annotation_data)
        
        if weight_len != 0 and weight_len != pval_len_m:
            sys.stderr.write("weight len can only be 0 or %d\n" % pval_len_m)
            sys.exit(1)
            
        if annotation_len != 0 and annotation_len != pval_len_m:
            sys.stderr.write("annotation len can only be 0 or %d\n" % pval_len_m)
            sys.exit(1)
        
        os.system("rm -rf %s*" % data_out)
        f_writer = open(data_out, "w")
        for i in xrange(0, pval_len_m):
            o_line = ""
            if weight_len > 0:
                o_line = "%s" % weight_data[i]
            else:
                o_line = "1"
            
            v_index = 0
            for j in xrange(0, pval_len_n):
                o_line = "%s %d:%s" % (o_line, v_index, pval_data[i][j])
                v_index += 1
            
            if annotation_len > 0:
                for j in xrange(0, len(annotation_data[i])):
                    o_line = "%s %d:%s" % (o_line, v_index, annotation_data[i][j])
                    v_index += 1
                
            f_writer.write("%s\n" % o_line)
        
        f_writer.close()
            
