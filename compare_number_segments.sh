#!/bin/bash
for ((a=2;a<=10;a++));
do
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor/Data
  rm -rf HLA-A-0201_padded/
  python blosum_encoding_padded.py HLA-A-0201 $a
  cd /home/sthodup1/comp_bio/MHC-Binding-Affinity-Predictor
  python YdiffCNN.py HLA-A-0201 True $a
done
