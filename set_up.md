<h1>1. Clone Megatron github</h1>

```console
$ mkdir prj && cd prj # make a new project folder
$ git clone https://github.com/sgohil3/Megatron-LM/tree/testProfiles && cd Megatron-LM
$ git checkout testProfiles # profiling script and conda env in this branch
```

<h1>2. Conda Env.</h1>
<h3>In Megatron-LM directory, </h3>

```console
$ conda env create --file=ncu_megatron_environment.yaml
$ conda activate Megatron-LM_pyEnv # create new enviro with given script and activate
$ cd .. && git clone (module file link)
$ source modules.sh # loads in the modules needed for SLURM
```

<h1>3. Transformer + Apex + Dataset Setup</h1>
<h2> Note: <i>update</i> the DIR_PROJECT variable in install_rest.sh with your project path</h2> 
<h3>In prj directory, </h3>

```console
$ source install_rest.sh
```

<h1>4. Profiling script </h1>
   <h3>In prj create a checkpoint folder</h3>

   ```console
   $ mkdir checkpoint && cd Megatron-LM
   ```
<ul> <h2> Update the variables for these scripts below</h2>
    <li> <h3> In "ap_dist_script.sh" update the "CHECKPOINT_PATH", "DATASET_PATH", and "MEGATRON_PATH" </h3></li>
    <li> <h3> In "slurm_loads.sh" update the ${USER} </h3> </li>
    <li> <h3> In "profilingScript.sh" update "PRJ_DIR" </h3> </li>
    <li> <h3> Note that to change with iteration to profile change "--profile-step-start/end=<iteration_id>" in profilingScript.sh </h3></li>
</ul>

```console
$ #call the program below to start profiling the Megatron-LM
$ ./profilingScript.sh NCU_SINGLE DP_TP_PP NumberOfGpus TP_Degree PP_Degree ITERATION_COUNT #replace NCU_SINGLE with NSYS for nsight systems and NCU_MULTI for nsight compute multiple gpus
$ ./profilingScript.sh NSYS DP_TP_PP NumberOfGpus TP_Degree PP_Degree ITERATION_COUNT #Iteration count can be any number for NCU_SINGLE/NSYS; its not required
```   
<h2> For Multi GPU with nsight compute, you must use the multi_gpu_profile_n_extract.py to profile the entire iteration. </h2>
<h2> Make sure to update the Launch Count and iterartion count to match that of the profiling script </h2>

```console
$ #call the program below to start profiling the Megatron-LM
$ python multi_gpu_profile_n_extract.py "DP_TP_PP NumberOfGpus TP_Degree PP_Degree" #this will profile all kernels at x Launch Count at t time and then recombine all the metrics into a single csv
```   

<h1>5. Data Extraction Scripts </h1>
<ul> <h3>
    <li> Once ncu profiling is complete, export the ncu_rep file in order to extract a csv</li>
</ul></h3>

```console
$ ncu -i ${report_name}.ncu-rep --csv --page raw --log-file ${final_path}/extracted_data.csv #all of extracted data into a csv file
$ ncu --query-metrics --log-file ${final_path}/all_metrics.txt #lists all available profiling metrics and their definitions
```

<ul> <h3>
    <li> Afterwards use the python script to plot the data as multiple plot or a single overlayed plot; it can also be use to extract a single metric into a .txt</li>
    <li> Don't forget to update the directory variables located at the top of the python script with the csv to read from </li>
    <li> don't forget to update what type of GPU is being used (ie: A100 vs v100) </li>
</ul></h3>

  ```console
 $ python plot_metric_data.py extract #use this command to extract the data from the .csv file 
 $ #NOTE: for load_plots/overlay it wants the file name in the format "metric_submetric.npz" ie: "dram__bytes_read_sum.npz"
 $ python plot_metric_data.py load_plots "metric_file_name_1" "...." "..." # given a list of file names plot each as a seperate graph 
 $ python plot_metric_data.py overlay "metric_file_name_1" "...." "..." #given a list of file names overlay each into a single graph
 $ #NOTE: for attribute give it the actual metric name ie: dram__bytes_read.sum
 $ python plot_metric_data.py attribute "metric_name_1" #gives a txt of the given metric (Note that this uses the actual metric+sub_metric given by the .csv file)
  ```

<h1>6. Random Errors that I ran into </h1>
<ul>
   <li><h2> Missing six/regex packages </h2></li>

  ```console
  $ pip install six 
  $ pip install regex 
  ```

   <li><h2> Missing transformer packages </h2></li>
   <ul>
      <li> make sure that all modules are loaded and that python packages are being installed at the right location </li>
  </ul>

   <ul>
      <li>if not make sure that OpenSSL and lang/Python/ are unloaded and then reactivate </li>
   </ul>

   <li><h2> Random freezing </h2></li>
   <ul>
      <li> comment out complile_helpers() in utils.py and run the make command manually </li>
      <li> update NCCL to the newest version available with the module system (Note: It will try to automatically update CUDA, make sure to revert it back) </li>
      <li> allocate more cores and set OMP_NUM_THREADS=$#_of_threads </li>
   </ul>

<li><h2> Still freezing </h2></li>
   <ul>
      <li> reduce the number of metrics/sections being profiled </li>
      <li> manually profile the metric you want one at a time, and then recombine them manually </li>
   </ul>

   <li><h2> CUDA could not allocate enough memory </h2></li>
   <ul>
      <li>try using SXM V100 or A100 </li>
      <li> use "watch -n 1 nvidia-smi" to check realtime utilization and memory usage of all GPUs </li>
      <li> use "HISTTIMEFORMAT='%x %X ' history" to check what time a command was run </li>
   </ul>
   
</ul>

<h1>7. NCCL Timeout Issue </h1>
<h2> I spent weeks trying to solve this issue/other workarounds. I had to apply all the above fixes first and create a python script which will split the kernel profiling to circumvent the timeout </h2>
<h2> If you are still facing this issue try these solutions below </h2>
<ul>
    <h3><li>Increase pytorch distributed timeout: At intialize.py:312, there is a torch.distributed.init_process_group. Increase the timeout to however long you need. This didn't work for me, but it did for others online</li></h3>
    <h3><li>Profile in batches and the recondense: There is a launch count number in both multi_gpu python profiling scirpt and the actual profiling script shell script. Decrease the launch count till you meet the timeout requirements. Don't forget to increase iteration to profile the entire application </h3></li> 
    <h3><li> I've noticed that for NCCL kernels that take a long time, they tend to freeze during kernel replay. My current theory is that since the NCCL kernel takes so long by the time a single NCCL kernel pass has completed, the other GPUs have moved on. As a result, the GPU we are profiling on tries to replay, but fails as the other GPUs don't answer the NCCL call.  </h3><li>
    <ul>
    <h3><li>Solution: Use application replay mode on the NCCL kernel and collect a minimum number of metrics and then manually recombine into a single csv</h3></li>
    </ul>
</ul>