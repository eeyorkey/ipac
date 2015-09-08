import time, sys, os

class AnnotationTask:
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
        # check
        need_solution_path = False
        if self.params.has_key("solution_path_needed") and self.params["solution_path_needed"]:
            need_solution_path = True
            self.cmd = "%s -p %s/solution_path.txt" % (self.cmd, self.params["work_dir"])
        
        if need_solution_path and self.params.has_key("plt_script") == False:
            sys.stderr.write("plt script is needed to show the solution path")
            sys.exit(1)
            
        f_out_model = "%s/model_%s" % (self.params["work_dir"], time.strftime("%Y%m%d_%H%M", \
                time.localtime()))
                
        if self.params.has_key("class_numbers"):
            print self.cmd % (self.ipac_version, self.f_p_val, f_out_model, self.params["class_numbers"])
            os.system(self.cmd % (self.ipac_version, self.f_p_val, f_out_model, self.params["class_numbers"]))
        else:
            print self.cmd % (self.ipac_version, self.f_p_val, f_out_model)
            os.system(self.cmd % (self.ipac_version, self.f_p_val, f_out_model))
        
        # begin to show the solution path
        if need_solution_path:
            plt_cmd = "%s %s/solution_path.txt %s/solution_path.png" % (self.params["plt_script"],\
                self.params["work_dir"], self.params["work_dir"])
            print plt_cmd
            os.system(plt_cmd)
