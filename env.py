'''
Global configurations
'''
# data loaders
# load p value
pvalue_data_loader = "data_loader_pval"
# load weight
weight_data_loader = "data_loader_weight"
# annotation data loader
annotation_data_loader = "data_loader_annotation"

# data combiner, which is used to transfer pval, weight and annotation
# file into the format of libSVM for ipac
data_combiner = "data_combiner.py"

'''
Task specific configurations
Note that: all data should be transfered into the libSVM data format
'''
# pval only
pval_task_cmd = "./bin/ipac_%s -x train -t %s -i 1 -k 200 -n 300 -o %s"
# pval and annotation
pval_annotation_task_cmd = "./bin/ipac_%s -x train -t %s -i 0 -k 200 -n 300 -o %s"

# script to draw solution path
solution_plt_script = "python2.7 plt.py"
