# Geographic Origin Prediction of Urban *Quercus rubra* using machine learning

This repository contains Python scripts used to predict the geographic origins of urban northern red oak (Quercus rubra) using genomic data and supervised machine learning, following the approach developed by *Hessenauer et al. (2026)*.

Models were trained using 738 individuals from natural populations across eastern North America and applied to 1,007 urban trees sampled in four Canadian cities.

## Scripts
All scripts were developed using Python 3.12 and are listed below in the order of the main analytical steps.

### **1. `make_input.py` — Data preparation**

Prepares the input data for machine-learning analyses by organizing SNP genotypes into a matrix, with individuals as rows and SNPs as columns, alongside geographic coordinates and population information.

**Arguments**

-genot: CSV file containing SNP genotypes.
id1: Sample ID column in the genotype file.
metad: CSV file containing sample metadata (including geographic coordinates).
id2: Sample ID column in the metadata file.
out: Output filename (without the .csv extension).

***Example*** 

`python make_input.py genotypes.csv Samples metadata.csv Sample_ID input`

### **2. `MLtraining_subsampSNPs_red_oak.py` — Model training and evaluation**

Trains and evaluates four machine-learning algorithms (Linear Model, Random Forest, Gradient Boosting, and KNN) using randomly selected SNP subsets from natural reference populations. The dataset is divided into training and validation sets (80 individuals for validation). Latitude and longitude are predicted separately, and model performance is assessed using the mean and standard deviation of geographic prediction errors (km). The script also outputs model performance metrics and diagnostic plots.

**Arguments**

- infile: Input CSV file containing SNP genotypes and geographic metadata.
- scaling: Feature scaling option (0 = no scaling; 1 = scaling).
- model: Machine-learning algorithm (Linear, RF, GBoost, or KNN).
- SNPsize: Number of SNPs in the input dataset when using all markers
- Ncpus: Number of CPU cores.

***Example*** 

`python MLtraining_subsampSNPs_red_oak.py input.csv 0 Linear 24796 4`

### **3. `getting_major_feature_red_oak.py` — SNP importance analysis**

Calculates permutation-based SNP importance using the selected Linear Model (LM). Importance is measured by the increase in root mean square error (RMSE) after randomly permuting each SNP. Scores are saved as CSV files for subsequent identification of informative markers.

**Arguments**

- file: Input CSV file containing SNP genotypes and geographic metadata.
- var: Geographic coordinate to predict (Latitude or Longitude).

***Example*** 

`python getting_major_feature_red_oak.py input.csv Latitude`

`python getting_major_feature_red_oak.py input.csv Longitude`

### **4. `ML_assignation_urban_tree_red_oak.py` — Urban tree origin prediction**

Trains the selected Linear Model using all 738 natural reference individuals and predicts the latitude and longitude of 1,007 urban trees. The input file contains the SNPs retained from the feature importance analysis.

**Arguments**

- infile: Input CSV file containing natural reference and urban individuals, including SNP genotypes and geographic metadata.*
- scaling: Feature scaling option (0 = no scaling; 1 = scaling).
- model: Machine-learning algorithm (Linear, RF, GBoost, or KNN).
- SNPsize: Number of SNPs in the input dataset, used in the output filename.
- Ncpus: Number of CPU cores.

***Example*** 

`python ML_assignation_urban_tree_red_oak.py input_selected_SNPs.csv 0 Linear 3539 4`

## Reference
Hessenauer, P., M. Zacharias, J. Prunier, et al. 2026. “ Harnessing Genomic Information for Identifying the Geographic Origin of Five North American Tree Species in Trade.” Evolutionary Applications 19, no. 8: e70295. [https://doi-org/10.1111/eva.70295.](https://doi.org/10.1111/eva.70295)
