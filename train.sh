#!/bin/sh
#SBATCH -c 1                # Request 1 CPU core
#SBATCH -p dl               # Partition to submit to (should always be dl, for now)
#SBATCH --mem=100G           # Request 100G of memory
#SBATCH -o logs/myoutput_%j.out  # File to which STDOUT will be written  (%j inserts jobid)
#SBATCH -e logs/myerrors_%j.err  # File to which STDERR will be written (%j inserts jobid)
#SBATCH --gres=gpu:1       

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-python-executable>"
    exit 1
fi

TRAIN_FILE="$1"


# Check if the file exists and is executable
if [ ! -f "$TRAIN_FILE" ] || [ ! -x "$TRAIN_FILE" ]; then
    echo "Error: $TRAIN_FILE is not an executable or does not exist."
    exit 1
fi


# Generate a unique name hash from the training job
TRAIN_FILE_NAME=$(basename $TRAIN_FILE)
TRAIN_FILE_HASH=$(md5sum $TRAIN_FILE_NAME | cut -d' ' -f1)

CURR_DATE=$(date +%Y-%m-%d)
CURR_TIME=$(date +%H:%M:%S)
mkdir -p train_jobs/$CURR_DATE

# Copy the python executable to the train_jobs directory with a uniquely generated name
cp $TRAIN_FILE train_jobs/$CURR_DATE/$CURR_TIME-$TRAIN_FILE_HASH.py

echo "Training with file $TRAIN_FILE and job hash $TRAIN_FILE_HASH"
python3 $TRAIN_FILE $TRAIN_FILE_HASH
