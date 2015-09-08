import sys, getopt, os
import env, ipac_task_scheduler
import data_loader_scheduler

short_opt = "hp:w:a:fd:"
long_opt = ["help", "pval=", "weight=", "annotation=", \
    "fig_solution_path", "work_dir="]
    
def print_help():
    print "TODO: please fillin the help information"
    
try:
    opts,args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
except getopt.GetoptError:
    print_help()
    sys.exit(1)

pvalue_file = ""
weight_file = ""
annotation_file = ""
work_dir = ""
solution_path_plt = False

for opt,value in opts:
    if opt in ("-h", "--help"):
        print_help()
        sys.exit(0)
    elif opt in ("-p", "--pval"):
        pvalue_file = value
    elif opt in ("-w", "--weight"):
        weight_file = value
    elif opt in ("-a", "--annotation"):
        annotation_file = value
    elif opt in ("-f", "--fig_solution_path"):
        solution_path_plt = True
    elif opt in ("-d", "--work_dir"):
        work_dir = value
    else:
        print "Unkown option: %s" % opt
        sys.exit(1)

if work_dir != "":
    if os.path.exists(work_dir):
        os.system("rm -rf %s" % work_dir)
    os.system("mkdir %s" % work_dir)
 
ipac = ipac_task_scheduler.IpacTaskScheduler()
ipac.init(pvalue_file, weight_file, annotation_file, \
    work_dir, solution_path_plt)
ipac.run_task()
   
