# FBG demodulated LPFG

## Description

This repository contains the data and code associated with the research paper titled "LPFG demoodulation using FBG sensor array and a self-attention fully-connected neural network". The paper presents a novel approach to demodulate long-period fiber grating (LPFG) sensors using a fiber Bragg grating (FBG) sensor array and a machine learning model. The key contributions of this work include:

- **Innovative Interrogation Method**: Utilization of a static FBG filter bank and machine learning techniques for industry-ready LPFG sensor interrogation.
- **Attention-Based Neural Network**: Implementation of a self-attention fully connected neural network to demodulate the LPFG based on features obtained by the reflection spectrum;
- **Transfer Learning**: Application of transfer learning from synthetic data training to calibrate machine learning models in optical sensing effectively.

The repository provides the necessary resources to replicate the study's findings, offering insights into the potential of FBGs as both sensors and filter banks for LPFG interrogation and the effectiveness of synthetic training transfer learning.


### Notebook description

- **1 - Data generation:** Generate synthetic data for model selection and training

- **2 - Model slection:** Choose hyperparameters based on Bayesian Search

- **3 - Retrain and save:** Retrain the model on new synthetic data

- **4 - Evaluation:** Evaluate the model

- **5 - Acquisition:** Make some measuremets using a HBM's BraggMeter



## Getting Started

### Dependencies

* Developed and tested in Windows 11 using Ubuntu 22.04 on WSL2
* To improve compatibility you could use the tf-wsl.yml file to create a conda env equal the one I used on WSL2 running Ubuntu 22.04.
* Measured LPFG spectra were collected using Anritsu MS9740B
* Measurements made using FS22DI Industrial BraggMETER from HBM.

### Executing a measurement

* For example of measuremnts, please see "5 - Acquisition.ipynb"

## The data

All data that was measured, i.e. real LPFG spectra or BraggMETER measurements are in /data folder. 

Synthetic datasets were too big to upload to this repository but similar data can be generated using *1 - Data generation* notebook.

## Authors

[Felipe Barino](mailto:felipe.barino@engenharia.ufjf.br)

## License

Licensed under the Apache License, Version 2.0
