# Code
The repository collects and openly releases the dataset and accompanying code for the results published in the following article: 
    
    Leonardo Peroni, Sergey Gorinsky. 2024. Quality of Experience in Video Streaming: Status Quo, Pitfalls, and Guidelines 

To reproduce the results, please follow the instructions provided in the readme.md files located inside the subfolders.

### Structure
We support the reproducibility recreates all figures and tables by running our Python code. 

### Artifact access
We structure the GitHub repository in folders dedicated to a figure or/and table. The subfolders hold all the data
and code required for generating the respective results.
### Hardware dependencies
There are no specific hardware requirements except for support of
Python 3.7. The machine used in our experiments is an Intel i7 with six cores, 2.6-GHz CPUs, 16-GB
RAM, and Windows 10.
### Software dependencies 
The requirements.txt file in the repository lists the software dependencies, with readme.md files in individual subfolders supplying any further details and clarifications.

### Datasets
In addition to the experiment-specific data in each subfolder, the repository includes dataset_120 of iQoE dataset<sup>1</sup> 
and Waterloo-IV dataset<sup>2</sup>.


### Installation
The reproducibility involves installation of Python 3.7 with libraries as described in the
requirements.txt. The experiments with baseline  model L require installation of BiQPS<sup>3</sup>. The experiments with baseline
model P require installation of ITU-T Rec. P.1203 Standalone Implementation<sup>4</sup> (we used the version 1.8.3 which is compliant with Python 3.7).

### Evaluation and expected results
The repository allocates a separate self-contained folder for reproducing the results of each figure and/or table, with readme.md files providing
specific instructions.

<sup>1</sup>https://github.com/Leo-rojo/iQoE_Dataset_and_Code

<sup>2</sup>https://dx.doi.org/10.21227/j15a-8r35

<sup>3</sup>https://github.com/TranHuyen1191/BiQPS

<sup>4</sup>https://github.com/itu-p1203/itu-p1203/tree/v1.8.3

### Requirements

We tested the code with `Python 3.7` with the following additional requirements that can be found in the `requirements.txt` file:

```
matplotlib==3.8.2
numpy==1.26.2
pandas==2.1.4
scikit_learn==1.3.2
scipy==1.11.4
seaborn==0.13.0
itu-p1203==1.8.3
```
