#!/bin/bash
for ((a=2;a<=10;a++));
do
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor/Data
  rm -rf HLA-A-0201_truncated/
  python blosum_encoding_truncated.py HLA-A-0201 $a
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor
  python YdiffDNN.py HLA-A-0201 False $a
done
