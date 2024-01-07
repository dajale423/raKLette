#!/bin/bash
#SBATCH -c 1                               # 1 core
#SBATCH -t 0-12:00                         # Runtime of 5 minutes, in D-HH:MM format
#SBATCH -p short                           # Run in short partition
#SBATCH -o hostname_%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like
#SBATCH --mem=1000M                         # Memory total in MiB (for all cores)
#SBATCH --mail-type=ALL                    # ALL email notification type
#SBATCH --mail-user=daniel_lee@g.harvard.edu  # Email to which notifications will be sent

rm slurm*
snakemake --unlock

command='snakemake --cluster "sbatch -c {resources.cpus_per_task} -t {resources.runtime} -p {resources.partition} --mem={resources.mem_mb}" --rerun-incomplete --latency-wait 30'

while getopts "j:r:c" opt; do
  case $opt in
     j)
       command="${command} -j $OPTARG" >&2
       ;;
     r)
       command="${command} --retries $OPTARG" >&2
       ;;
     c)
       command="${command} --use-conda" >&2
       ;;
     *)
       ;;
  esac
done

# echo $command
eval $command