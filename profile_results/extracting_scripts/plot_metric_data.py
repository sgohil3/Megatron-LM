import csv
import sys
import numpy as np
import datetime
import os
import pickle
import matplotlib.pyplot as plt
import mplcursors

#update this for your computer
RESULTS_DIR = "path_to_results_directory" #directory where our results will be located
L2_SIZE_MB  = 40
L1_SIZE_MB  = (192/1024) #nvidia uses base 2

def extract_metric_data():
    intialize       = 0
    metric_to_data  = {} #dictionary of metric name to its list of data
    metric_to_units = {} #dictionary of metric name to its unit of measurement
    metric_list     = []

    with open("extracted_data.csv", newline='') as csv_metrics: #go through our exported report and generate an explanation for each metric found
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
        
def plot_data(metric_to_data, metric_to_units, metric_to_plot):
    
    float_time      = np.array(metric_to_data["gpu__time_duration.sum"], dtype=float)#/1000 #convert to seconds (use this if you wish to use time as an x-axis instead)
    # sprint(metric_to_units["gpu__time_duration.sum"])
    cumulative_time = np.cumsum(float_time) #this is will become our x-axis       
    kernel_id       = np.array(range(0,len(metric_to_data["gpu__time_duration.sum"])))

    for curr_metric in metric_to_plot: #create a graph for each metric
        if("device__attribute" in curr_metric): #don't make a graph for a device__attribute metric
            continue

        metric_name     = curr_metric[:curr_metric.find(".")]
        submetric_name  = curr_metric[(curr_metric.find(".") + 1):]
        curr_Units      = np.array(metric_to_units[curr_metric]) # units ie: bytes/ratio/seconds etc..
        curr_data       = metric_to_data[curr_metric]
        float_data      = np.array([float(curr_data[i]) for i in range(len(curr_data))])
        
        submetric_name = submetric_name.replace('.', "_") #replace . with _ to ensure readablity
        submetric_name = submetric_name.replace('/', "_") #replace / with _ to ensure we_don't accidentally make a new folder

        file_name = f"{RESULTS_DIR}{metric_name}_{submetric_name}" 
        os.makedirs(os.path.dirname(file_name), exist_ok=True) #check that the file_directory that we want to write to exists
        # print(file_name, curr_Units)
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
    cache_plot  = 1

    all_files   = os.listdir(RESULTS_DIR)
    figure, ax  = plt.subplots(figsize=(40,24))
    ax.set_title(str(overlay_plots) + ' Per Kernel')
    figure.canvas.manager.set_window_title(str(overlay_plots))

    for file_name in all_files:
        for entries in overlay_plots: #double for loop is dumb find better way
            if entries in str(file_name): #only display plot if substring is present in our list
                
                print(entries, file_name)
                npzfile         = np.load(f"{RESULTS_DIR}{file_name}")
                float_data      = npzfile['arr_0']
                cumulative_time = npzfile['arr_1']
                kernel_id       = npzfile['arr_2']
                curr_Units      = str(npzfile['arr_3'])
                metric_name     = file_name[:file_name.find(".")]
                
                ax.set_xlabel('Time [msecond]')
                ax.set_ylabel(curr_Units)
                ax.scatter(cumulative_time, float_data, label=metric_name, s=5) 
                
                if(cache_plot and 'dram' in metric_name and ('%' not in curr_Units or '/s' not in curr_Units)):
                    cache_plot = 0
                    ax.plot(cumulative_time, ([L1_SIZE_MB + L2_SIZE_MB/2] * len(cumulative_time)), label="l1 + min l2 size", linewidth=0.5, color="green")
                    ax.plot(cumulative_time, ([(L2_SIZE_MB + L1_SIZE_MB)] * len(cumulative_time)), label="l1 + max l2 size", linewidth=0.5, color="black")

                
    ax.legend(loc='upper center')            
    mplcursors.cursor(figure, hover=True)#adds the ability to hover to dictate what point we are looking at
    plt.show()

    return None

def extract_metric_explanations():
    result_metrics      = open("final_available_metrics.txt", "w")
    success_metric      = []
    explanation_dict    = create_explanation_dict()

    with open("extracted_data.csv", newline='') as csv_metrics: #go through our exported report and generate an explanation for each metric found
        available_metrics = csv.reader(csv_metrics, delimiter=',')
        for metric_row in available_metrics:
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

def create_explanation_dict(): #creates a dictionary from metric name to explanation
    tot_metrics         = open("all_metrics.txt", "r")
    explanation_dict    = {}
    all_lines           = tot_metrics.readlines()

    for entry in all_lines: #create a dictionary mapping metric names to metric explanations
        explanation_dict[entry[:entry.find(' ')]] = entry[entry.find(' '):]
    tot_metrics.close()
    
    return explanation_dict

def extract_attribute(metric_to_data, metric_to_units, metric): # Use this if you want to extract a single data point with its Kernel ID
    data = metric_to_data[metric]                               # It is super useful if you want to get Device Attributes
    unit = metric_to_units[metric]

    with open(f"{metric}_extracted.txt", 'w') as file_name:
        file_name.writelines("Kernel ID " + str(i) + ": " + data[i] + "\n" for i in range(len(data)))

    return None

def case(funct_to_call, plot_names): #switch statement for what functions to execute
    if funct_to_call == "explain": # creates a mapping for metric name from report to explaination
        return extract_metric_explanations()
    elif funct_to_call == "extract": # extracts metric data from report and stores it 
        metric_to_data, metric_to_units, metric_to_plot = extract_metric_data()
        metric_to_plot.remove("launch__func_cache_config") 
        return plot_data(metric_to_data=metric_to_data, metric_to_units=metric_to_units, metric_to_plot=metric_to_plot[11:])
    elif funct_to_call == "load_plots": # generates a plot for each metric called
        return load_data(plot_names)
    elif funct_to_call == "device_attribute": # extract a single metric and it's data in text form
        metric_to_data, metric_to_units, metric_to_plot = extract_metric_data()
        metric_to_plot.remove("launch__func_cache_config")
        return extract_attribute(metric_to_data=metric_to_data, metric_to_units=metric_to_units, metric=plot_names[0])
    elif funct_to_call == "overlay": # plot all metrics listed into one graph
        return load_overlayed_data(plot_names)
    else:
        print("Error: add explain, extract, (load_plots, #, #) after script name")
          
def main():
    function_call = sys.argv[1]
    plot_names    = sys.argv[2:] 
    case(function_call, plot_names)
    
if __name__ == "__main__":
    main()