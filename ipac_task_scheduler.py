import sys, os
from tasks import *
import env
import data_loader_scheduler
import data_combiner

class IpacTaskScheduler:
    def __init__(self):
        self.f_pval = ""
        self.f_weight = ""
        self.f_annotation = ""
        self.f_workdir = ""
        self.plt_solution_path = False
        self.params = {}
        
    def init(self, f_pval, f_weight, f_annotation, \
        f_workdir, solution_path_plt):
        if f_workdir == "":
            f_workdir = "./"
        
        self.f_pval = f_pval
        self.f_weight = f_weight
        self.f_annotation = f_annotation
        self.f_workdir = f_workdir
        self.plt_solution_path = solution_path_plt
        self.params = {}
        
    def run_task(self):
        # check input
        if self.f_pval == "":
            sys.stderr.write("pval file should exist\n")
            sys.exit(1)
            
        data_scheduler = data_loader_scheduler.DataLoaderScheduler()
        pval_data = data_scheduler.get_data(self.f_pval, env.pvalue_data_loader)
        
        ipac_version = "single"
        self.params = {"work_dir":self.f_workdir}
        cmd_append = ""
        if len(pval_data[0]) > 1:
            ipac_version = "multiple"
            self.params["class_numbers"] = len(pval_data[0])
            cmd_append = " -l %d"
        
        weight_data = []
        annotation_data = []
        
        if self.f_weight != "":
            weight_data = data_scheduler.get_data(self.f_weight, env.weight_data_loader)
            
        if self.f_annotation != "":
            annotation_data = data_scheduler.get_data(self.f_annotation, env.annotation_data_loader)
            
        data = {"pval":pval_data, "weight":weight_data, "annotation":annotation_data}
        data_cb = data_combiner.DataCombiner()
        merged_data_path = os.path.join(self.f_workdir, "combined_data")
        data_cb.combine(data, merged_data_path)
        
        ''' register ipac tasks'''
        if self.f_annotation == "":
            p_val_task = pval_task.PvalTask(env.pval_task_cmd + cmd_append, ipac_version, \
                merged_data_path, self.params)
            p_val_task.run()
        else:
            if self.plt_solution_path:
                self.params["solution_path_needed"] = True
                self.params["plt_script"] = env.solution_plt_script
            
            ann_task = annotation_task.AnnotationTask(env.pval_annotation_task_cmd + cmd_append, \
            ipac_version, merged_data_path, self.params)
            ann_task.run()