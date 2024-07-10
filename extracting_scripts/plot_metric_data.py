import csv
import sys
import numpy as np
import datetime
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import mplcursors
import platform
# how to run:
# python plot_metric_data.py extract        #use this to extract the profiling data from the csv, update the CSV_FILE variable below
# python plot_metric_data.py explain        #use this to extract all the metrics from your csv and link them with an explanation for what it represents
# python plot_metric_data.py load_plots 'file_name.npz' 'file_name_2.npz' ...  # use this command to plot each given metric in a seperate graph
# python plot_metric_data.py attribute 'metric_name' 'metric_name' ...         # use this command with the exact metric name given by nvidia to extract a single metric from our csv
# python plot_metric_data.py overlay 'file_name.npz' 'file_name_2.npz' ...     # use this command to plot all the metrics into a single graph


#update these global program variables for your data extraction
FORWARD_DP_1     = 453
BACKWARD_DP_1    = 1524

FORWARD_DP_2     = 454
BACKWARD_DP_2    = 1525

FORWARD_TP_2     = 513
BACKWARD_TP_2    = 1634

#set this to which parallelism we are profiling
Forward_Count    = FORWARD_TP_2
Backward_Count   = BACKWARD_TP_2

#switch when using V100/A100
USING_V100       = False
CSV_FILE         = "/imec/scratch/dtpatha/gohil01/test2/prj/extracting_scripts/extracted_csv/DP_2_A100_extracted_data.csv"
RESULTS_DIR      = "/imec/scratch/dtpatha/gohil01/test2/prj/extracting_scripts/saved_results/" #directory where our results will be located
EXTRACT_DIR      = "/imec/scratch/dtpatha/gohil01/test2/prj/extracting_scripts"

A100_L2_SIZE_MB  = 40
V100_L2_SIZE_MB  = 6
L1_SIZE_MB       = (192/1024) #nvidia uses base 2
        
def parse_and_store_data(metric_to_data, metric_to_units, metric_to_plot):
    
    float_time      = np.array(metric_to_data["gpu__time_duration.sum"], dtype=float)#/1000 #convert to seconds (use this if you wish to use time as an x-axis instead)
    # sprint(metric_to_units["gpu__time_duration.sum"])
    cumulative_time = np.cumsum(float_time) #this is will become our x-axis       
    kernel_id       = np.array(range(0,len(metric_to_data["gpu__time_duration.sum"])))

    for curr_metric in metric_to_plot: #create a graph for each metric
        if("device__attribute" in curr_metric or "pmsampling" in curr_metric): 
            continue #string types not enabled so filter out device__atribute and pmsampling

        metric_name     = curr_metric[:curr_metric.find(".")]
        submetric_name  = curr_metric[(curr_metric.find(".") + 1):]
        curr_Units      = np.array(metric_to_units[curr_metric]) # units ie: bytes/ratio/seconds etc..
        curr_data       = metric_to_data[curr_metric]
        print(curr_metric)
        float_data      = np.array([float(curr_data[i]) for i in range(len(curr_data))])
        print(" success")
        submetric_name = submetric_name.replace('.', "_") #replace . with _ to ensure readablity
        submetric_name = submetric_name.replace('/', "_") #replace / with _ to ensure we_don't accidentally make a new folder

        file_name = f"{RESULTS_DIR}{metric_name}_{submetric_name}" 
        os.makedirs(os.path.dirname(file_name), exist_ok=True) #check that the file_directory that we want to write to exists
        np.savez(file_name, float_data, cumulative_time, kernel_id, curr_Units)

    return None

def load_data(plot_name):
    all_files = os.listdir(RESULTS_DIR)

    for file_name in all_files:
        for entries in plot_name: #double for loop is dumb find better way
            if entries in str(file_name): #only display plot if substring is present in our list
                
                print(entries, file_name)
                npzfile         = np.load(f"{RESULTS_DIR}{file_name}")
                float_data      = npzfile['arr_0']
                cumulative_time = npzfile['arr_1']
                kernel_id       = npzfile['arr_2']
                curr_Units      = str(npzfile['arr_3'])
                metric_name     = file_name[:file_name.find(".")]
                
                figure, ax      = plt.subplots(figsize=(40,24))
                figure.canvas.manager.set_window_title(metric_name)
                ax.set_xlabel('Time [msecond]')
                ax.set_ylabel(curr_Units)
                ax.set_title(metric_name + ' Per Kernel')
                ax.scatter(cumulative_time, float_data, label=metric_name) 
                plt.show()

                mplcursors.cursor(figure, hover=True)#adds the ability to hover to dictate what point we are looking at
    
    plt.waitforbuttonpress()

    return None

def load_overlayed_data(overlay_plots):
    all_files   = os.listdir(RESULTS_DIR)
    all_figures = []
    color_list  = ["darkblue", "seagreen", "darkorange", "pink"]  #may need to add more if more graphs are overlayed 
    cache_plot  = 0
    n_cols      = 0

    all_files   = os.listdir(RESULTS_DIR)
    figure, ax  = plt.subplots(figsize=(14,20))
    figure.canvas.manager.set_window_title(str(overlay_plots))
    delin_ymax  = 0
    
    for file_name in all_files:
        for entries in overlay_plots: #double for loop is dumb find better way
            if entries in str(file_name): #only display plot if substring is present in our list
                
                # print(entries, file_name)
                npzfile         = np.load(f"{RESULTS_DIR}{file_name}")
                float_data      = npzfile['arr_0']
                cumulative_time = npzfile['arr_1']
                kernel_id       = npzfile['arr_2']
                curr_Units      = str(npzfile['arr_3'])
                metric_name     = file_name[:file_name.find(".")]

                if (metric_name == "l1tex__m_xbar2l1tex_read_bytes_sum" and USING_V100 == False): #if grabbing l1tex_readbytes need to offset by shared memoyr bypass
                    npzfile     = np.load(f"{RESULTS_DIR}sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass_sum.npz")
                    offset_data = npzfile['arr_0']
                    float_data  = float_data - offset_data  
                  
                if curr_Units == "Gbyte": #convert everything in Gbyte to Mbyte
                    float_data = float_data * 1024
                    curr_Units = "Mbyte"

                if curr_Units == "Kbyte": #convert everything in Kbyte to Mbyte
                    float_data = float_data / 1024
                    curr_Units = "Mbyte"

                if curr_Units == "sector": #convert everything in Sector to Mbyte
                    float_data = (float_data *32) / (2**20)
                    curr_Units = "Mbyte"
                
                print(str(metric_name) +": " + str(np.sum(float_data[:Backward_Count])) + " Mbyte")
                print(cumulative_time[Backward_Count])
                
                # ax.set_title("Data Parallelism PCIe Flow" + ' Per Kernel')
                # ax.set_ylim(((5.257305088045188e-05, 16695.235277326454))) #DP_1 read y range
                # ax.set_ylim((0.00031076061352851205, 18.322881266539834))  #TP_2 Pcie y range
                # ax.set_ylim((1.4112458261483375e-05, 936.7925816356225))   #TP_2 Write y range
                # ax.set_ylim((0.0004834378475409073, 7168.783201689118))    #TP_2 Read y range
                if(metric_name == "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass_sum"):
                    label = "Shared Mem read from L2" 
                elif(metric_name == "l1tex__m_xbar2l1tex_read_bytes_sum"):
                    label = "L1 read from L2"
                elif(metric_name == "dram__bytes_read_sum"):
                    label = "L2 read from DRAM"
                    ax.set_title("V100 Memory Flow(Read)" + ' Per Kernel')
                elif(metric_name == "dram__bytes_write_sum"):
                    label = "L2 write to DRAM" 
                    ax.set_title("V100 Memory Flow(Write)" + ' Per Kernel')
                elif(metric_name == "l1tex__m_l1tex2xbar_write_bytes_sum"):
                    label = "L1 write to L2" 
                elif(metric_name == "lts__t_sectors_srcunit_ltcfabric_sum"):
                    label = "L2 Fabric"
                    ax.set_title("Total Bytes Transfered Over L2 Fabric" + ' Per Kernel')
                else:
                    label = metric_name
                
                forward_pass_end_stamp = cumulative_time[Forward_Count] 
                ax.set_xlabel('Time [msecond]')
                ax.set_ylabel(curr_Units) 
                plt.axvline(x=forward_pass_end_stamp, color="red") #plot delination line between forward/backwards pass
                ax.scatter(cumulative_time[:Backward_Count], float_data[:Backward_Count], label=label, s=5, c=color_list[n_cols]) 
                n_cols += 1
                plt.yscale("log", base=2)

                # forward_pass_end_stamp = Forward_Count 
                # ax.scatter(kernel_id[:Backward_Count], float_data[:Backward_Count], label=label, s=5, c=color_list[n_cols])
            
                if(('lts__t_sectors_srcunit_ltcfabric_sum' in metric_name or 'dram' in metric_name) and ('%' not in curr_Units or '/s' not in curr_Units)):
                    cache_plot = 1
                
          
    mplcursors.cursor(figure, hover=True)#adds the ability to hover to dictate what point we are looking at
    if(cache_plot):
        #use this for A100
        if(USING_V100 == False):
            ax.plot(cumulative_time[:Backward_Count], ([A100_L2_SIZE_MB/2] * Backward_Count), label="min l2 size", linewidth=3, color="purple")
            ax.plot(cumulative_time[:Backward_Count], ([(A100_L2_SIZE_MB)] * Backward_Count), label="max l2 size", linewidth=3, color="black")
            n_cols += 2
        else:# use this for V100
            ax.plot(cumulative_time[:Backward_Count], ([(V100_L2_SIZE_MB)] * Backward_Count), label="l2 size", linewidth=3, color="black")
            n_cols += 1
        
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=n_cols, fancybox=True, shadow=True, prop={'size': 10}, markerscale=6)
    
    print("limit is: ", ax.get_ylim())
    plt.show(block=True)
    figure.waitforbuttonpress()

    return None

def extract_attribute(metric_to_data, metric_to_units, metric): # Use this if you want to extract a single data point with its Kernel ID
    data = metric_to_data[metric]                               # It is super useful if you want to get Device Attributes
    unit = metric_to_units[metric]

    with open(f"{EXTRACT_DIR}/attributes/{metric}_extracted.txt", 'w') as file_name:
        file_name.writelines("Kernel ID " + str(i) + ": " + data[i] + " " + unit + "\n" for i in range(len(data)))

    return None

def extract_metric_explanations():
    result_metrics      = open("final_available_metrics.txt", "w")
    success_metric      = []
    explanation_dict    = create_explanation_dict()

    with open(CSV_FILE, newline='') as csv_metrics: #go through our exported report and generate an explanation for each metric found
        all_metrics = csv.reader(csv_metrics, delimiter=',')
        for metric_row in all_metrics:
            for metric in metric_row: 

                sub_metric_index =  metric.find(".")
                sub_metric       = metric[sub_metric_index:]
                metric           = metric[:sub_metric_index]
                if metric in explanation_dict:
                    if (metric not in success_metric): #if we have a match and not already added 
                        success_metric.append(metric)
                        result_metrics.write("\n\n" + (metric) + "          " + explanation_dict[metric])
                        result_metrics.write(sub_metric + " ")
                    else: #if already added then only put down submetric
                        result_metrics.write(sub_metric + " ")
            break

    result_metrics.close()

def extract_metric_data():
    intialize       = 0
    metric_to_data  = {} #dictionary of metric name to its list of data
    metric_to_units = {} #dictionary of metric name to its unit of measurement
    metric_list     = []

    with open(CSV_FILE, newline='') as csv_metrics: #go through our exported report and generate an explanation for each metric found
        available_metrics = csv.reader(csv_metrics, delimiter=',')
        for metric_row in available_metrics:
            curr_metric_index = 0
            for metric in metric_row: 

                if(intialize == 0): # only on the first row, intialize our dictionary and list
                    metric_to_data[metric] = []
                    metric_list.append(metric)
                elif(intialize == 1):# on second row, intailize our unit dictionary
                    metric_to_units[metric_list[curr_metric_index]] = metric
                else: #on every other row record our data into a list
                    metric_to_data[metric_list[curr_metric_index]].append(metric)
                curr_metric_index += 1

            intialize += 1

    return metric_to_data, metric_to_units, metric_list

def create_explanation_dict(): #creates a dictionary from metric name to explanation
    tot_metrics         = open("all_metrics.txt", "r")
    explanation_dict    = {}
    all_lines           = tot_metrics.readlines()

    for entry in all_lines: #create a dictionary mapping metric names to metric explanations
        explanation_dict[entry[:entry.find(' ')]] = entry[entry.find(' '):]
    tot_metrics.close()
    
    return explanation_dict


def case(funct_to_call, plot_names): #switch statement for what functions to execute
    if funct_to_call == "explain": # creates a mapping for metric name from report to explaination
        return extract_metric_explanations()
    
    elif funct_to_call == "extract": # extracts metric data from report and stores it 
        metric_to_data, metric_to_units, metric_to_plot = extract_metric_data()
        metric_to_plot.remove("launch__func_cache_config") 
        # metric_to_plot.remove("pmsampling:dramc__read_throughput.avg.pct_of_peak_sustained_elapsed")

        file_name = f"{EXTRACT_DIR}metric_dictionary.pkl" 
        with open(file_name, 'wb') as f:
            print("dumping pickle")
            pickle.dump([metric_to_data, metric_to_units, metric_to_plot], f)
        
        return parse_and_store_data(metric_to_data=metric_to_data, metric_to_units=metric_to_units, metric_to_plot=metric_to_plot[11:])
    
    elif funct_to_call == "load_plots": # generates a plot for each metric called
        return load_data(plot_names)
    
    elif funct_to_call == "attribute": # extract a single metric and it's data in text form
        file_name = f"{EXTRACT_DIR}metric_dictionary.pkl"
        with open(file_name, 'rb') as f:
            saved_dicts     = pickle.load(f)
            metric_to_data  = saved_dicts[0]
            metric_to_units = saved_dicts[1]
            metric_to_plot  = saved_dicts[2]
        
        return extract_attribute(metric_to_data=metric_to_data, metric_to_units=metric_to_units, metric=plot_names[0])
    
    elif funct_to_call == "overlay": # plot all metrics listed into one graph
        return load_overlayed_data(plot_names)
    
    else:
        print("Error: add explain, extract, (load_plots, #, #) after script name")
        return None
          
def main():
    function_call = sys.argv[1]
    plot_names    = sys.argv[2:] 
    case(function_call, plot_names)
    
if __name__ == "__main__":
    main()