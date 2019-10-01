# Single Cell Deep Learning
Unsupervised representation learning of single-cell RNA-sequencing profiles using generative deep learning models

*Note: This repository is just a snapshot of the research/work done and is not a finished project.*

## Structure
- ```framework``` - models and framework code (dataset, experiment)
- ```experiments``` - training experiments
- ```scripts``` - R scripts for analysis/interpretation of results
- ```slurm``` - slurm shell scripts for training jobs on Pittsburgh Bridges Supercomputer
- ```data``` - single-cell data (not included on Github)

## Implemented Models
- Deterministic Autoencoder
- Variational Autoencoder (w/ support for Bernoulli or Mean Guassian output distributions)
- [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644) (Makhzani et al)
- [Unsupervised Clustering Autoencoder](https://arxiv.org/abs/1611.02648) (experimental, Dilokthanakul et al)
- Generative Adversarial Network
- Wasserstein GAN
