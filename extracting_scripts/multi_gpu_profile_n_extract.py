import csv
import sys
import os
import datetime
import pickle

NUM_OF_ITERATIONS = 10
LAUNCH_COUNT      = 450
REPORT_DIR="/imec/scratch/dtpatha/gohil01/test2/prj/Megatron-LM/profile_results/TEST_EXTRACT/"

def condense_reports(): #this will condense all the reports in the file, make sure there aren't extra reports you don't want to be extracted
    i=0
    final_report     = open("final_available_metrics.csv", "w")
    all_files        = os.listdir(REPORT_DIR)
    report_list      = []
    
    for files in all_files: # for each report, extract the raw metrics into a csv file
        if ".ncu-rep" in files:
            extracted_report_name = str(i) + "_extracted_data.csv"
            report_list.append(extracted_report_name)
            command = "ncu -i " + REPORT_DIR + str(files) + " --csv --page raw --log-file " + str(extracted_report_name)
            print(command)
            # os.system(command) #run the ncu command
            i += 1
    
    i = 0  
    print("starting condensing")      
    for CSV_FILE in report_list: #read each csv file, and recombine it into one large one
        with open(CSV_FILE, newline='') as csv_metrics:
            j = 0
            entire_file = csv.reader(csv_metrics, delimiter=',')
            for metric_row in entire_file:
                if i != 0 and (j == 0 or j == 1): # for every report but the first one, skip the first two lines
                    j += 1
                    continue
                else:
                    if (j == 0 or j == 1):
                        # print(metric_row)
                        final_report.write(str(metric_row)[1:-1])
                    else:
                        metric_row[0] = (LAUNCH_COUNT * i) + (j - 2) #update the profiled kernel ID to be in order
                        final_report.write(str(metric_row)[1:-1])
                        final_report.write('\n')
                    j += 1

def main():
    function_call = "./profilingScript.sh NCU_MULTI " + str(sys.argv[1]) + " " 
    print(function_call)
    # for i in range(NUM_OF_ITERATIONS):
    #     os.system(function_call + str(i)) # call our profiling script multiple times to profile in batches of 1000
    condense_reports()
    
if __name__ == "__main__":
    main()