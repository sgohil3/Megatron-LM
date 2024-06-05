
#enter the results tab and keep note of where to save our results
PROF_DIR="/imec/scratch/dtpatha/gohil01/test2/prj/Megatron-LM/profile_results/DP_1_1_TP_1_1_PP_1_1"
REPORT_NAME=$1
#update the variable below to change what to output from the report
metric_list="gpu__time_duration.sum,gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum"

echo "start extraction"
ncu -i ${PROF_DIR}/${REPORT_NAME}.ncu-rep --csv --page raw --log-file extracted_data.csv 
# --metrics ${metric_list}
echo "finished exporting"
ncu --query-metrics --log-file all_metrics.txt
echo "finish querying"
python plot_metric_data.py explain
python plot_metric_data.py extract
echo "finish extraction"