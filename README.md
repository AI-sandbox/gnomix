# XGMix: Local-Ancestry Inference With Stacked XGBoost

This repository includes a python implemenation of XGMix, a gradient boosting tree-based local-ancestry inference (ancestry deconvolution) method. 

XGMIX.py can be used in two ways:

- training a model from scratch using provided training data or 
- loading a pre-trained XGMix model (see **Pre-Trained Models** below)

In both cases the models are used to infer local ancestry for provided query data.

## Dependencies
The dependencies are listed in *requirements.txt*. Assuming [pip](https://pip.pypa.io/en/stable/) is already installed, they can be installed via
```
$ pip install -r requirements.txt
```
When using the program for training a model, [BCFtools](http://samtools.github.io/bcftools/bcftools.html) must be installed and available in the PATH environment setting.

## Usage

### When Using Pre-Trained Models
XGMIX.py loads and uses pre-trained XGMix models to predict the ancestry for a given *<query_file>* and chromosome number. 

To execute the program with a pre-trained model run:
```
$ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <path_to_model> 
```

where 
- <*query_file*> is a .vcf or .vcf.gz file containing the query haplotypes which are to be analyzed (see example in the **demo_data/** folder)
- <*genetic_map_file*> is the genetic map file (see example in the **demo_data/** folder)
- <*output_basename*>.msp.tsv and <*output_basename*>.fb.tsv is where the predictions are written (see details in **Output** below and an example in the **demo_data/** folder)
- <*chr_nr*> is the chromosome number
- <*phase*> is either True or False corresponding to the intent of using the predicted ancestry for phasing (see details in **Phasing** below and in the **XGFix/** folder)
- <*path_to_model*> is a path to the model used for predictions (see **Pre-trained Models** below)

### When Training a Model From Scratch
XGMix.py loads data from the *<reference_file>* 

To execute the program when training a model run:
```
$ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <reference_file> <sample_map_file>
```

where the first 5 arguments are described above in the pre-trained setting and 
- <*reference_file*> is a .vcf or .vcf.gz file containing the reference haplotypes (in any order)
- <*sample_map_file*> is a sample map file matching reference samples to their respective reference populations

The program uses these two files as input to our simulation algorithm (see **pyadmix/**) to create training data for the model.

### Calibration
XGmix output probabilities might not reflect the true confidence / accuracy of the predictions. By setting to True calibration when training a new model, Isotonic Regression is used to match the predicted probabilities to calibrated probabilities. For example, in a calibrated model, predictions with a probability 80% will be correct 80% of the time.

### Advanced Options
More advanced configuration settings can be found in *config.py*. 
They include general settings, simulation settings and model settings. More details are given in the file itself.

## Output

### *<output_basename>*.msp.tsv
The first line is a comment line, that specifies the order and encoding of populations, eg:
#Sub_population order/code: golden_retriever=0 labrador_retriever=1 poodle poodle_small=2

The second line specifies the column names, and every following line marks a genome position.

The first 6 columns specify
- the chromosome
- interval of genetic marker's physical position in basepair units (one column represents the starting point and one the end point)
- interval of genetic position in centiMorgans (one column represents the starting point and one the end point)
- number of *<query_file>* SNP positions that are included in interval

The remaining columns give the predicted reference panel population for the given interval. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes) and therefore the total number of columns in the file is 6 + 2*(number of genotypes)

### *<output_basename>*.fb.tsv
The first line is a comment line, that specifies the order of the populations, eg:
#reference_panel_population:	AFR	EUR	NAT

The second line specifies the column names, and every following line marks a genome position.

The first 4 columns specify
- the chromosome
- mean of genetic marker's physical position in basepair units
- mean of genetic position in centiMorgans
- genetic marker index

The remaining columns represent the query hapotypes and reference panel population and each line markes the estimated probability of the given genome position coming from the population. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes)*(number of reference populations) and therefore the total number of columns in the file is 6 + 2*(number of genotypes)*(number of reference populations).

### Model and simulated data
When training a model, the resulting model will be stored in **./models**. That way it can be re-used for analyzing another dataset.
The model's estimated accuracy is logged along with a confusion matrix which is stored in **./models/analysis**.
The program simulates training data and stores in **./generated_data**. To automatically remove the created data when training is done,
set *rm_simulated_data* to True in *config.py*. Note that in some cases, the simulated data can be re-used for training with similar settings. 
In those cases, not removing the data and then setting *run_simulation* to False will re-use the previously simulated data which can save a lot of time and compuation.

## Pre-Trained Models

Pre-trained models will soon be available for download from [XGMix-models](https://github.com/AI-sandbox/XGMix-models).

When making predictions, the input to the model is an intersection of the pre-trained model SNP positions and the SNP positions from the <query_file>. That means that the set of positions that's only in the original training input is encoded as missing and the set of positions only in the <query_file> is discarded. When the script is executed, it will log the intersection-ratio as the performance will depend on how much of the original positions are missing. When the intersection is low, we recommend using a model trained with high percentage of missing data.

The models are trained on hg build 37 references from the following biogeographic regions: *Subsaharan African (AFR), African Hunter and Gatherer (AHG), East Asian (EAS), European (EUR), Native American (NAT), Oceanian (OCE), South Asian (SAS), and West Asian (WAS)* and labels and predicts them as 0, 1, .., 7 respectively.

## Phasing


![Visualization of the process](XGFix/figures/XGFix.gif)


Accurate phasing of genomic data is crucial for human demographic modeling and identity-by-descent analyses. It has been shown that leveraging information about an individual’s genomic ancestry improves performance of current phasing algorithms. XGFix is a method that uses local Ancestry Inference (LAI) to do exactly that. If you suspect your data might be badly phased (often the case when reference panel is small and/or diverse), we recommend using this option. See the **XGFix/** folder for more details. 

![Local Ancestry for Phasing Error Correction](XGFix/figures/laipec_resized.png)
Sequenced haplotypes phased with a phasing software (left). LAI used to label haplotypes with ancestry predictions and phasing errors become evident (center). Phasing error correction using LAI is applied to correct phasing errors (right).

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" must contact the [Stanford Office of Technology Licensing](https://otl.stanford.edu/) for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to that effect.

## Cite

#### When using this software, please cite: Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A., "XGMix: Local-Ancestry Inference With Stacked XGBoost," International Conference on Learning Representations Workshops (ICLR, 2020, Workshop AI4AH).

https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1

```
@article{kumar2020xgmix,
  title={XGMix: Local-Ancestry Inference With Stacked XGBoost},
  author={Kumar, Arvind and Montserrat, Daniel Mas and Bustamante, Carlos and Ioannidis, Alexander},
  journal={International Conference of Learning Representations Workshops, AI4AH},
  year={2020}
}
```

#### You can also include its companion paper: Montserrat, D.M., Kumar, A., Bustamante, C. and Ioannidis, A., "Addressing Ancestry Disparities in Genomic Medicine: A Geographic-aware Algorithm," International Conference on Learning Representations Workshops (ICLR, 2020, Workshop AI4CC).

https://arxiv.org/pdf/2004.12053.pdf

```
@article{montserrat2020addressing,
  title={Addressing Ancestry Disparities in Genomic Medicine: A Geographic-aware Algorithm},
  author={Montserrat, Daniel Mas and Kumar, Arvind and Bustamante, Carlos and Ioannidis, Alexander},
  journal={International Conference of Learning Representations Workshops, AI4CC},
  year={2020}
}
```



