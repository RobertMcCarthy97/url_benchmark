#!/bin/bash -l
#SBATCH --job-name=ProtoPush
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=gpu
# Request 2 gpus
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=16

# specify the walltime e.g 20 mins
#SBATCH -t 20:20:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

module load anaconda
source activate drqv2

# command to use
# python3 pretrain.py agent=proto obs_type=pixels suite=multigoal bullet_task=push
# python3 finetune.py agent=proto obs_type=pixels suite=multigoal bullet_task=push include_r_intr=true snapshot_ts=0
python3 finetune.py agent=icm obs_type=pixels suite=minitouch bullet_task=Pushing-v0 include_r_intr=true snapshot_ts=0 update_encoder=true

# sbatch --partition=gpu train.sh
