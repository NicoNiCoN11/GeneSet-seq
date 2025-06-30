#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 1:00:00
#SBATCH --mem 4G

source /home/jiguo/denovo_rpe1_scrnaseq/venvs/rna/bin/activate
python /home/jiguo/denovo_rpe1_scrnaseq/analysis_scripts/preprocess.py