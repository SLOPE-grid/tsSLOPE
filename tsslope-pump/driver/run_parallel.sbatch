#!/bin/bash -l
#SBATCH --job-name=tsacopf_cnf_array
#SBATCH --output=logs/tsacopf_sol_%j.out
#SBATCH --error=logs/tsacopf_sol_%j.err
#SBATCH -N 1
#SBATCH -c 112
#SBATCH -p pbatch
#SBATCH -A hiop
#SBATCH -t 12:00:00

set -euo pipefail
JOBID="${SLURM_JOB_ID:-nojobid}"
echo "Job started at $(date) on $(hostname) (JOBID=$JOBID)"
start_time=$(date +%s)

#module load julia

# avoid nested threading oversubscription
export JULIA_NUM_THREADS=56
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

### for old versions
#julia run_all_samples_CNN.jl
#julia driver_parallel.jl

#export folder_start=0
#julia driver.


mkdir -p logs

folder_start=0 julia driver_7k.jl >> "logs/group001_025_${JOBID}.log" 2>&1 &
pid1=$!

folder_start=25 julia driver_7k.jl >> "logs/group026_050_${JOBID}.log" 2>&1 &
pid2=$!

folder_start=50 julia driver_7k.jl >> "logs/group051_075_${JOBID}.log" 2>&1 &
pid3=$!

folder_start=75 julia driver_7k.jl >> "logs/group076_100_${JOBID}.log" 2>&1 &
pid4=$!

#######################################
# Wait for all to finish
#######################################
fail=0

wait "$pid1"
status1=$?
echo "Task 1 exit code: $status1"
if [[ $status1 -ne 0 ]]; then
  fail=1
fi

wait "$pid2"
status2=$?
echo "Task 2 exit code: $status2"
if [[ $status2 -ne 0 ]]; then
  fail=1
fi

wait "$pid3"
status3=$?
echo "Task 3 exit code: $status3"
if [[ $status3 -ne 0 ]]; then
  fail=1
fi

wait "$pid4"
status4=$?
echo "Task 4 exit code: $status4"
if [[ $status4 -ne 0 ]]; then
  fail=1
fi

if [[ $fail -ne 0 ]]; then
  echo "One or more tasks failed."
  exit 1
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "All tasks completed successfully at $(date)"
printf "Elapsed time: %02d:%02d:%02d (%d seconds)\n" \
  $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) "$elapsed"
  
exit 0



