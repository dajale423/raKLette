#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-12:00
#SBATCH -p short
#SBATCH --mem=40G                           # Memory total in MB (for all cores)
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=daniel_lee@g.harvard.edu
#SBATCH -o run_model_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e run_model_%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like
python raklette_gene.py
