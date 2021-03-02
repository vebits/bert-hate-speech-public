#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --gres=gpu:2
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64000
#SBATCH --job-name="bert-hstuning"
#SBATCH --output=out/bert-test/bert-test-run03.out
#SBATCH --mail-user=vebjorni@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "We are using $SLURM_GPUS_ON_NODE GPUs per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load GCC/7.3.0-2.30
module load icc/2018.3.222-GCC-7.3.0-2.30
module load ifort/2018.3.222-GCC-7.3.0-2.30
module load OpenMPI/3.1.1
module load impi/2018.3.222
module load Python/3.6.6
module load CUDA/9.0.176

python run_lm_hstuning.py \
  --bert_model bert-large-uncased \
  --do_lower_case \
  --do_train \
  --train_file data/all_all_data_gold.txt \
  --output_dir models/bert-large \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
