#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH --partition=interactive
#SBATCH -c 6
#SBATCH --mem=64GB
#SBATCH --job-name=dev_node
#SBATCH --output=runs/dev_node_%j.out

# Thin, GPU-less interactive dev node held for 1 day so Shahaf can SSH in
# (PyCharm remote / code reading). 6 vCPU / 64GB -- enough headroom for
# PyCharm's indexer over the gaussianformer tree on NFS. No GPU: heavy
# compute still goes through separate sbatch jobs.
#
# Having an active job on the node is what grants SSH access to it.
# sleep holds the allocation until walltime; scancel <JOBID> releases early.

cat <<BANNER
=========================================================
dev_node up.  job=$SLURM_JOB_ID  node=$(hostname)
SSH target: $(hostname)   (24h walltime)
Release early: scancel $SLURM_JOB_ID
=========================================================
BANNER

sleep infinity
