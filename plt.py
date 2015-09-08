import matplotlib
matplotlib.use('Agg')

import sys, os
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    sys.stderr.write("%s input_data output_figure" % sys.argv[0])
    sys.exit(1)
    
f_input_data = sys.argv[1]
f_output_fig = sys.argv[2]

def load_data(f_name):
    f_reader = open(f_name)
    ret_data = []
    for each_line in f_reader:
        s_list = each_line.strip().split(" ")
        row_data_x = []
        row_data_y = []
        for i in range(1, len(s_list)):
            sub_list = s_list[i].split(":")
            row_data_x.append(int(sub_list[0]))
            row_data_y.append(float(sub_list[1]))
            
        ret_data.append([row_data_x, row_data_y])
    f_reader.close()
    
    return ret_data
	
def get_plt_data(fig_data):
	ret_data = []
	
	max_index = 0
	for each_row in fig_data:
		for each_x in each_row[0]:
			if max_index < each_x:
				max_index = each_x
	
	max_index = max_index + 1	
	for i in range(0, len(fig_data)):
		ret_data.append([0]*max_index)
		
	for i in range(0, len(fig_data)):
		for j in range(0, len(fig_data[i][0])):
			d_index = fig_data[i][0][j]
			d_value = fig_data[i][1][j]
			ret_data[i][d_index] = d_value
			
			if j == 0:
				continue
			last_index = fig_data[i][0][j-1]
			last_value = fig_data[i][1][j-1]
			
			mid_len = d_index - last_index - 1
			if mid_len > 0:
				ret_data[i][(last_index+1):d_index] = [last_value] * mid_len
				
		last_index = fig_data[i][0][-1]
		last_value = fig_data[i][1][-1]
		if max_index > last_index + 1:
			ret_data[i][(last_index+1):max_index] = [last_value] * (max_index - last_index - 1)
	
	return ret_data
    
fig_data = load_data(f_input_data)    
plt_data = get_plt_data(fig_data)

max_index = len(plt_data[0])
plot_x = [0] * max_index
for i in range(0, max_index):
	plot_x[i] = i

plt.plot(plot_x, plt_data[0], '-')
plt.hold(True)
plt.grid(True)
for i in range(1, len(plt_data)):
    plt.plot(plot_x, plt_data[i], '-')
plt.title('Solution path')
plt.xlabel('Iterations')
plt.ylabel('Coefficients')
plt.savefig(f_output_fig)
