#!/bin/bash -l

########
### SLURM PARAMETERS
########

#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#SBATCH -o ./job.out
#SBATCH -e ./job.err



########
### LOAD MODULES AND/OR SET CONDA-ENV.
########

# ...



########
### LAUNCH PROGRAM
########

# find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $1}'`
echo IP address $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)
echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "$SLURM_NODEID Launching python script"

srun python3 PRGRM_NAME.py --nnodes 32 --ntasks 4 --ngpus 4 --ip_address $ip1 > prog.out
