#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name rank2k_SL             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note that SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-fe.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-fe.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 16GB

# Ensure the logs directory exists
mkdir -p logs

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/workspace/miniconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate fe

# Running the job
start=`date +%s`

# python hpo.py
python run_train_test.py --json "ranked_2k.json" --layer "sl" --full True --kd True --pixel True
# python run_train_test.py --json "random_2k.json" --layer "l5" --full True --kd False --pixel False
# python run_train_test.py --json "random_2k_every_4th.json" --layer "l5" --full True --kd False --pixel False
end=`date +%s`
runtime=$((end-start))

# delete outputs/ directory and all its contents
rm -rf outputs/


echo Job execution complete.
echo Runtime: $runtime
