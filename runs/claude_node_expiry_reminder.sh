#!/bin/zsh
#SBATCH --time=00:01:00
#SBATCH -c 1
#SBATCH --mem=256MB
#SBATCH --begin=2026-06-13T09:57:00
#SBATCH --job-name=REUP_CLAUDE_NODES_EXPIRE_TOMORROW
#SBATCH --output=runs/claude_node_reminder_%j.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=shahaf.valenciale@mail.huji.ac.il

# One-shot reminder. Sits in PD with reason=BeginTime until 2026-06-13 09:57,
# then runs, triggering a SLURM BEGIN mail to Shahaf. The job-name IS the
# subject line of that mail -- the actual signal. The body below also lands
# in the .out file as a fallback.
#
# Trigger time chosen for ~34 h lead before both claude_node jobs (30652003,
# 30652017) expire ~19:33 local on 2026-06-14.

cat <<'EOF'
=========================================================
REMINDER: claude_node sbatch jobs expire ~19:33 local 2026-06-14.

Check current state:
  squeue -u $USER -n claude_node -o "%i %N %T %M %L"

Re-submit (verify SESSION + WORKDIR for each):
  sbatch --export=ALL,SESSION=gaussianformer-encoding,WORKDIR=/cs/labs/tomhope/shahaf_levy/gaussianformer /cs/labs/tomhope/shahaf_levy/claude_node.sh
  sbatch --export=ALL,SESSION=tri_level,WORKDIR=/cs/labs/tomhope/shahaf_levy/tri_level /cs/labs/tomhope/shahaf_levy/claude_node.sh

Attach to the new job:
  srun --jobid=<NEW_JOBID> --overlap --pty tmux attach -t <SESSION>
  # inside tmux: claude --resume

Do NOT scancel the old jobs until the new ones are attached and verified.
=========================================================
EOF
