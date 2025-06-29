# How-The-Brain-Moves
This is the repository for all documents related the TFM: *How the Brain Moves: Understanding Motion-Based Brain States Using Deep Learning Techniques*, written by Joshua Tapper. In this report, neural decoding was performed on Local Field Potential (LFP) data, using three different model architectures: Convolutional Neural Network, Convolutional Neural Network into Long Short-Term Memory, and a parallel Convolutional Neural Network and Long Short-Term Memory. The importance of input features was then extracted using three methods: SHAP values, Integrated Gradients, and ablation.

## Notable Techniques Used
$${\color{gray}Models \space were \space trained \space and \space analyzed \space using \space Python \space 3.11.13, \space TensorFlow \space 2.18.0, \space and \space Keras \space 3.8.0.}$$

* Spectrogram tansformation and cleaning
* Bad channel detection and deletion
* Adam optimizer
* Early stopping
* Learning rate reduction
* CNN model
* CNN into LSTM model
* Parallel CNN and LSTM model
* Convolutional encoder
* PCA
* Grid search
* SHAP DeepExplainer and KernelExplainer
* Integrated Gradients
* Ablation testing  

## Document Description

* **TrainModels.ipynb** - Used to train and evaluate the models described in the report.
* **FeatureExtraction.ipynb** - Used to analyze the models and generate many of the figures used in the report.
* **IG_values.ipynb** - Used to acquire the Integrated Gradients on trained models.
* **SHAP_values.py** - Used to acquire the SHAP values using the KernelExplaienr.
* **Archive.ipynb** - A collection of code cells that provide a proof of concept of the testing and experimentation performed before the work described in the final report.

## Contact

Any issues, questions or comments are welcome at:

* Email: jtappeta214@alumnes.ub.edu

### BibTex for the Repository

```
@misc{TapperRepository,
title={How The Brain Moves},
url={https://github.com/jtapper13/How-The-Brain-Moves/},
note={GitHub repository},
author={Joshua Tapper},
  year={2025}
}
```

### BibTex for the Report

```
@mastersthesis{TapperThesis,
    author = {Joshua Tapper},
    title = {How the Brain Moves: Understanding Motion-Based Brain States Using Deep Learning Techniques},
    school = {Universitat de Barcelona},
    year = {2025}
}
```
