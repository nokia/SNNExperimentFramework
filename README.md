# SNNFrameWork

Repository used to store and track the SSN (short for Siamese Neural Network) tool.
This tool main purpose is to easily experiemnt and train [Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)
with different configurations and different features.

This tool has been used to obtain the results presented in the following papers:

- [Optimizing Predictive Analytics in 5G Networks Through Zero-Trust Operator-Customer Cooperation](https://e-archivo.uc3m.es/bitstream/handle/10016/39458/Optimizing_NFV-SDN_2023_ps.pdf?sequence=1)
presented at [NFV-SDN 2023](https://nfvsdn2023.ieee-nfvsdn.org) and awarded as best paper.
- [ATELIER: service tailored and limited-trust network analytics using cooperative learning](),
the link will be updaed once the paper will be published and available.

The Repository is separated into two main sections:

- DataGeneration, this folder contains all the tools and information used to
produce the datasets used furter for the training of the Neural Network model
through the SNN tool.
- SNN This folder contains the actual python code of the SNN tool, plus all the
configuration objects and post-processing scripts used for both the pubblications
previously mentioned

## Datasets

Please notice that due to the size of the datasets used in this projects, such
files are not available directly in the repository.
But both the dataset generated with the DataGeneration tool (Dkr5G) and the
one generated as outcome from the all the training cycles are available on [Zenodo](https://zenodo.org)

Please refer to the 'reproducibility' section in the wiki for details on how
to download and use the mentioned datasets.
