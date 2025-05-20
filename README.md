# Graph Distributional Signals for Regularization in Graph Neural Networks

This repository contains the code for our IEEE TSPIN accepted paper.

## Table of Contents
- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements
To install the required dependencies, refer to the `environment.yml` file.

## Reproducing Results
We use GCN as the baseline model, implemented in the GCN file. To reproduce the results in Table 3, run the following command:

```bash
python train_GCN_RLoss.py 
python train_GCN_Preg.py 
python train_GCN_Lreg.py 


```
These scripts correspond to different regularization methods, referred to as R-GCN, P-GCN, and L-GCN, respectively. Among them, R-GCN is our proposed method.
