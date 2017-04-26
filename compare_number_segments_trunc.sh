#!/bin/bash
for ((a=1;a<=30;a++));
do
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor/Data
  rm -rf H-2-Db_truncated/
  python blosum_encoding_truncated.py H-2-Db $a
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor
  python YdiffCNN.py H-2-Db False $a
done
