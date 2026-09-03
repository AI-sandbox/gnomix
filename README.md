

# Scalable high resolution ancestry deconvolution for genomic data

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-brightgreen)

<br>


![Visualization of the process](https://github.com/AI-sandbox/gnomix/blob/main/doc/fig/gnomix_diagram.png)

This repository includes a Python implementation of Gnomix, a fast, scalable, and accurate local ancestry method. See [demo](demo.ipynb).

Gnomix can be used in two ways:

- training a model from scratch using reference training data or 
- loading a pre-trained Gnomix model (see **Pre-Trained Models** below)

In both cases the models are used to infer local ancestry on provided query data that has already been phased (using a program like beagle, shapeit, or eagle) and pre-processed to have the same sites as the reference training samples on the same strand, or if a pre-trained model is used instead see **Pre-Trained Models** below for requirements.

## Installation and Dependencies

To install the software, clone the repository and create a Python environment:

```bash
git clone https://github.com/AI-sandbox/gnomix
cd gnomix
conda create -n gnomix python=3.14
conda activate gnomix
```

You may replace `3.14` with any supported Python version between `3.9` and `3.14`. However, we strongly recommend using Python `3.13` or `3.14`, as they provide substantially faster runtimes than earlier Python releases.

The dependencies are listed in *requirements.txt*. Assuming [pip](https://pip.pypa.io/en/stable/) is already installed, they can be installed via:

```bash
pip install -r requirements.txt
```

## Usage

### Best Practices

For recommended workflows and common tuning options, see **[Best Practices](./gnomix-best-practices.md)**. It covers input preparation tips (build consistency, liftover, and imputation) and guidance for adjusting key hyperparameters via the config file. We recommend using **MAF of .01, no LD pruning, phased, and only biallelic snps**. Before training Gnomix, we also recommend filtering the reference panel VCF for common variants, such as variants with minimum allele count ≥ 20. The same filtering criteria should then be applied to the query dataset. After filtering both files, take the intersection of retained variants; this intersection is the final set of variants to use for Gnomix training.

We recommend looking at different configuration files, see **[Config Files](./configs/README.md)**. In particular, we recommend different configurations depending on whether working with whole genome or array data, human genomes or plant genomes, and Tracts analyses or snp focused (GWAS, PRS, PCA, F-statistics) analyses on the downstream side.



### When Training a Model From Scratch

To execute the program when training a model run:
```
$ python3 gnomix.py <query_file> <output_folder> <chr_nr> <phase> <genetic_map_file> <reference_file> <sample_map_file>
```

where the first 4 arguments are described above in the pre-trained setting and 
- <*genetic_map_file*> is the genetic map file. It's a .tsv file with 3 columns; chromosome number, SNP physical position and SNP genetic position. There should be no headers unless they start with "#". See example in the **demo/data/** folder.
- <*reference_file*> is a .vcf or .vcf.gz file containing the reference haplotypes (in any order)
- <*sample_map_file*> is a sample map file matching reference samples to their respective reference populations

The program uses these two files as input to our simulation algorithm (see **pyadmix/**) to create training data for the model. Also, note that when running inference on the trained models, the <*query_file*> needs to have the same build as the genetic map used to train the model. (For instance, in the case of humans, it is build37 or build38)


### Advanced Options
More advanced configuration settings can be found in *config.yaml*. 
They include general settings, simulation settings and model settings. More details are given in the file itself. If training a model from scratch you can also pass an alternative config file as the last argument:

```
$ python3 gnomix.py <query_file> <output_folder> <chr_nr> <phase> <genetic_map_file> <reference_file> <sample_map_file> <config_file>
```

If no config is given, the program uses the default (*config.yaml*). The config file has advanced training options. Some of the parameters are
- verbose (bool) - verbosity (default True)
- simulation:
  - run: (bool) - whether to run simulation or not, can be skipped if previously done (default True)
  - path: (path) - # where to store the simulated data, if run is False this is where the simulation data will be sought, default is <output_folder>/generated_data/
  - r_admixed (float,positive) - number of simulated admixed individuals generated when training the model = r_admixed x size of sample map (number of reference samples). The default is 1. Set it lower if memory is an issue. (To overcome memory constraints a minor allele frequency filter can also be used to remove very rare variants.)
  - splits: must contain proportion for train1, train2 and optionally validation. If validation ratio is 0, validation is not performed.
  - generations indicates the total specturem of generations since admixture to simulate, not critical
  - rm_data (bool) - whether to remove simulated data after training (to conserve disk space). It is set to false if run is False. Default False.
- model:
  - name (string) - model's name: default is "model"
  - inference (string) - 4 possible options - best / fast / large / default. "best" uses random string kernel base + xgboost smoother and is recommended for array data. "fast" uses logistic regression base + crf smoother. "large" uses logistic regression + convolutional smoother and is good for large datasets for which memory requirements are an issue. "default" uses logistic regression base + xgboost smoother and on whole genome has nearly the same accuracy as "best," but with much faster runtime.
  - window_size_cM (float, positive) -  size of window in centiMorgans, use larger windows if snp density is lower e.g. genotype data vs. sequence (default .5)
  - smooth_size (int, positive) - number of windows to be taken as context for smoother (default 75)
  - context_ratio (float between 0 and 1) - context of base model windows (default .5)
  - windowed_loading (bool) - set to True to read train1 one window at a time, or False to load it fully into memory (default False)
  - retrain_base (bool) - retrain base models using both train1 and train2 once smoother is trained, validation data for a final base model (default True)
  - calibrate (bool) - applies calibration on output probabilities (default False)
  - n_cores (int, positive) - how many units of cpu to use (default is maximum), reduce if you are on a shared cluster and using only a subset of nodes
- inference:
  - bed_file_output: generate files for each individual that show the run length encoding of their ancestry segments (default False)
  - snp_level_inference: output ancestry inference for each marker of the query file (default False)
  - visualize_inference: create pictures showing the ancestry segments colored along each individual's chromosomes using Tagore (default False)

#### More model combinations

For more base + smoother combinations one can edit the *gnomix.py* file in the following way:

import the base model of choice from src/base/model e.g., 

```python
from src.Base.models import LogisticRegressionBase
```

import the smoother of choice from src/smooth/model e.g., 

```python
from src.Smooth.models import XGB_Smoother
```

and then, in the train_model() function in initilize the Gnomix object with the imported models:
 
```python
model = Gnomix(
	...,
	base = LogisticRegressionBase,
	smooth = XGB_Smoother,
	...
)
```

For pre-trained models see **[Demo Pretrained](./demo-pretrained.md)**

## Output

The results (including predictions, trained models and analysis) are stored in the *<output_folder>*.

### Inference

The inference is written to two files, one for a single ancestry estimates for each marker (qery_results.msp) and one for probability estimates for each ancestry at each marker (query_results.fb). Below, we describe the both files in more detail.

#### query_results.msp

In the query_results.msp file, the first line is a comment line, that specifies the order and encoding of populations, eg:
#Sub_population order/code: golden_retriever=0 labrador_retriever=1 poodle poodle_small=2

The second line specifies the column names, and every following line marks an interval on the genome.

The first 6 columns specify
- the chromosome
- interval of genetic marker's physical position in basepair units (one column represents the starting point and one the end point)
- interval of genetic position in centiMorgans (one column represents the starting point and one the end point)
- number of *<query_file>* SNP positions that are included in interval

The remaining columns give the predicted reference panel population for the given interval. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes) and therefore the total number of columns in the file is 6 + 2*(number of genotypes)

#### query_results.fb

In the query_results.fb file, the first line is a comment line, that specifies the order of the populations, eg:
#reference_panel_population:	AFR	EUR	NAT

The second line specifies the column names, and every following line marks an interval on the genome.

The first 4 columns specify
- the chromosome
- mean of genetic marker's physical position in base pair units
- mean of genetic position in centiMorgans
- genetic marker index

The remaining columns represent the query hapotypes and reference panel population and each line markes the estimated probability of the given genome position coming from the population. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes)*(number of reference populations) and therefore the total number of columns in the file is 6 + 2*(number of genotypes)*(number of reference populations).

#### query_results.lai **(BETA)**

The query_results.lai is an optional output that includes the inferred ancestry label for each marker in the query file. Please note that this feature is in beta stage and therefore the program does not export this file unless *snp_level_inference* is set to *True* in the *config.yaml* file.

The first line of the output file is a comment line, that specifies the order and encoding of populations, eg:
#Sub_population order/code: golden_retriever=0 labrador_retriever=1 poodle poodle_small=2
just like in the msp file.

The second line specifies the column names, and every following line marks a genome position.

The first column indicates the physical position of the SNP and the remaining columns give the predicted reference panel population for the given interval. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes) and therefore the total number of columns in the file is 1 + 2*(number of genotypes).

#### query_file_phased.vcf

When using Gnofix for phasing error correcting (See Phasing below), the inference above will be performed on the query haplotype phased by Gnofix. These phased haplotypes will then also be exported to query_file_phased.vcf in the *<output_folder>*/ folder.

### Visualization
To visualize the local ancestry output along the chromosome using [tagore](https://pypi.org/project/tagore/#usage) for plotting, use the visualize_inference True option in the config file.

### Model
When training a model, the resulting model will be stored in *<output_folder>/models*. That way it can be re-used for analyzing another dataset.
The model's estimated accuracy is logged along with a confusion matrix which is stored in *<output_folder>/models/analysis*.

### Simulated data
The program simulates training data and stores it in *<output_folder>/generated_data*. To automatically remove the created data when training is done,
set *rm_simulated_data* to True in *config.yaml*. Note that in some cases, the simulated data can be re-used for training with similar settings. 
In those cases, not removing the data and then setting *run_simulation* to False will re-use the previously simulated data which can save a lot of time and compuation.

## Phasing

![Depiction of the process](https://github.com/AI-sandbox/gnomix/blob/main/src/Gnofix/figures/XGFix.gif)

Accurate phasing of genomic data is crucial for human demographic modeling and identity-by-descent analyses. It has been shown that leveraging information about an individual’s genomic ancestry improves performance of current phasing algorithms. Gnofix is a method that uses local ancestry inference to do exactly that. If you suspect your data might have phasing errors (generally the case unless trio phasing was possible), we recommend using this option <*phase*> as True. See the **gnofix/** folder if interested in more details on the algorithm. 

![Local Ancestry for Phasing Error Correction](https://github.com/AI-sandbox/gnomix/blob/main/src/Gnofix/figures/laipec_resized.png)
Sequenced haplotypes phased with a phasing software (left). LAI is used to label haplotypes with ancestry predictions and phasing errors become evident (center). Phasing error correction using LAI is applied to correct phasing errors (right). Small numbers of phasing errors do not, however, impact the correct association of a variant with an ancestry, and so are typically only a visual nuisance.

## Calibration
To ensure that Gnomix outputs probability estimates that reflect it's true confidence and accuracy, we recommend using calibration. We use Isotonic Regression to map the predicted probabilities to calibrated probabilities where the latter are more likely to have predictions with a confidence of X% correct matching their actual X% frequency of being correct in practice.



## License

**NOTICE**: This software is available free of charge for academic research use only. Commercial users and for profit companies or consultants can use the features present in this software by contacting [Galatea Bio](https://www.galatea.bio/), to which [Stanford Office of Technology Licensing](https://otl.stanford.edu/) has exclusively licensed technology used in this package. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to this effect.

## Cite

#### When using this software, please cite: 
### Helgi Hilmarsson, Arvind S Kumar, Miriam Barrabes, Richa Rastogi, Carlos D Bustamante, Daniel Mas Montserrat, Alexander G Ioannidis: "Scalable high resolution ancestry deconvolution for genomic data"

https://www.nature.com/articles/s41467-026-75391-0

```
@article{hilmarsson2026scalable,
  title={Scalable high resolution ancestry deconvolution for genomic data},
  author={Hilmarsson, Helgi and Kumar, Arvind S and Barrab{\'e}s, M{\'\i}riam and Rastogi, Richa and Bustamante, Carlos D and Montserrat, Daniel Mas and Ioannidis, Alexander G},
  journal={Nature Communications},
  year={2026},
  doi={10.1038/s41467-026-75391-0},
  publisher={Nature Publishing Group UK London}
}
