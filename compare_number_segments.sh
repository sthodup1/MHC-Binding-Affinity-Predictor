#!/bin/bash
for ((a=11;a<=30;a++));
do
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor/Data
  rm -rf H-2-Db_padded/
  python blosum_encoding_padded.py H-2-Db $a
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor
  python YdiffCNN.py H-2-Db True $a
done
