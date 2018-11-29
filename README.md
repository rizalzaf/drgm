# Distributionally Robust Graphical Models
This repository is a code example for the NeurIPS 2018 paper: 
[Distributionally Robust Graphical Models](https://papers.nips.cc/paper/8055-distributionally-robust-graphical-models).

### Abstract

In many structured prediction problems, complex relationships between variables are compactly defined using graphical structures. The most prevalent graphical prediction methods---probabilistic graphical models and large margin methods---have their own distinct strengths but also possess significant drawbacks. Conditional random fields (CRFs) are Fisher consistent, but they do not permit integration of customized loss metrics into their learning process. Large-margin models, such as structured support vector machines (SSVMs), have the flexibility to incorporate customized loss metrics, but lack Fisher consistency guarantees. We present adversarial graphical models (AGM), a distributionally robust approach for constructing a predictor that performs robustly for a class of data distributions defined using a graphical structure. Our approach enjoys both the flexibility of incorporating customized loss metrics into its design as well as the statistical guarantee of Fisher consistency. We present exact learning and prediction algorithms for AGM with time complexity similar to existing graphical models and show the practical benefits of our approach with experiments.

# Setup

The source code is written in [Julia](http://julialang.org/) version 0.6.X.

### Dependency

The following packages are needed to run the experiments:
* MAT.jl : for reading MATLAB data file.
* HDF5.jl : for storing results in HDF file format.
* Clp.jl : for solving linear program.

To install the packages, from Julia console, type `Pkg.add('PkgName')`.

### Experiments

Two files are provided for running the emotion intensity prediction: 
* `main_emotion_cv.jl` :
train an AGM model for the prediction task and evaluate the performance. 

* `crf_emotion_cv.jl` :
train a CRF model for the prediction task and evaluate the performance. 

To change the loss metric used, please edit the files above. Three loss metric are possible: zero-one, absolute, and squared loss metrics.


### Dataset

The datasets for the experiment is taken from  'Structured output ordinal regression for dynamic facial emotion intensity prediction' paper (ECCV 2010) by Minyoung Kim and Vladimir Pavlovic. They publish their codes and datestes [here](https://github.com/RWalecki/DOC-Toolbox). Please download the `ck.mat` file from the repository and place it inside `emotion` folder before running the experiments.

# Citation (BibTeX)
```
@inproceedings{fathony2018distributionally,
  title = {Distributionally Robust Graphical Models},
  author = {Fathony, Rizal and Rezaei, Ashkan and Bashiri, Mohammad Ali and Zhang, Xinhua and Ziebart, Brian},
  booktitle = {Advances in Neural Information Processing Systems 31},
  editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
  pages = {8353--8364},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/8055-distributionally-robust-graphical-models.pdf}
}

```
# Acknowledgements 
This work was supported, in part, by the National Science Foundation under Grant No. 1652530, and by the Future of Life Institute (futureoflife.org) FLI-RFP-AI1 program.
