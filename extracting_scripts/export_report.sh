
#enter the results tab and keep note of where to save our results
PROF_DIR="/imec/scratch/dtpatha/gohil01/test2/prj/Megatron-LM/profile_results/DP_1_1_TP_1_1_PP_1_1"
EXTR_DIR="/imec/scratch/dtpatha/gohil01/test2/prj/Megatron-LM/profile_results/extracting_scripts/"

REPORT_NAME=$1
#update the variable below to change what to output from the report
metric_list="gpu__time_duration.sum,gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum"

echo "start exporting"
ncu -i ${PROF_DIR}/${REPORT_NAME}.ncu-rep --csv --page raw --log-file ${EXTR_DIR}extracted_data.csv 
# --metrics ${metric_list}
echo "finished exporting/start query"
ncu --query-metrics --log-file ${EXTR_DIR}all_metrics.txt
echo "finish querying/start extraction"
python ${EXTR_DIR}plot_metric_data.py explain
python ${EXTR_DIR}plot_metric_data.py extract
echo "finish extraction"