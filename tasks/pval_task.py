import time, sys, os

class PvalTask:
    def __init__(self, cmd, ipac_version, p_val_file, params):
        if params.has_key("work_dir") == False:
            sys.stderr.write("work dir should be specified for the pval task\n")
            sys.exit(1)
            
        if cmd == "" or p_val_file == "" or params["work_dir"] == "":
            sys.stderr.write("either cmd, pval_file or ouput_dir is empty\n")
            sys.exit(1)
            
        self.cmd = cmd
        self.ipac_version = ipac_version
        self.f_p_val = p_val_file
        self.params = params
        
    def run(self):
        f_out_model = "%s/model_%s" % (self.params["work_dir"], time.strftime("%Y%m%d_%H%M", \
                time.localtime()))
                
        if self.params.has_key("class_numbers"):
            print self.cmd % (self.ipac_version, self.f_p_val, f_out_model, self.params["class_numbers"])
            os.system(self.cmd % (self.ipac_version, self.f_p_val, f_out_model, self.params["class_numbers"]))
        else:    
            print self.cmd % (self.ipac_version, self.f_p_val, f_out_model)
            os.system(self.cmd % (self.ipac_version, self.f_p_val, f_out_model))
