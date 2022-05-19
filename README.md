<img align="left" src="doc/fig/G-Nomix.png" width=18.7% height=18.7%> 

# High Resolution Ancestry Deconvolution for Next Generation Genomic Data 
<br>


![Visualization of the process](doc/fig/gnomix_diagram.png)

This repository includes a python implementation of G-Nomix, a fast, scalable, and accurate local ancestry method. See [demo](demo.ipynb).

G-Nomix can be used in two ways:

- training a model from scratch using reference training data or 
- loading a pre-trained G-Nomix model (see **Pre-Trained Models** below)

In both cases the models are used to infer local ancestry on provided query data that has already been phased (using a program like beagle, shapeit, or eagle) and pre-processed to have the same sites as the reference training samples on the same strand, or if a pre-trained model is used instead see **Pre-Trained Models** below for requirements. 

## Installation and Dependencies

To install the software, navigate to the desired folder and enter in the command line interface:
```
git clone https://github.com/AI-sandbox/gnomix
cd gnomix
```

The dependencies are listed in *requirements.txt*. Assuming [pip](https://pip.pypa.io/en/stable/) is already installed, they can be installed via
```
$ pip install -r requirements.txt
```

The combined runtime for the cloning and the dependency installation should be around 2 minutes on a normal laptop.

The software has been tested in Python 3.7.4 on the following operating systems:
- Linux: Ubuntu 18.04.5
- macOS: Monterey (12.0.1)

## Usage

### When Using Pre-Trained Models
gnomix.py loads and uses a pre-trained G-Nomix model to predict the ancestry for a given *<query_file>* and a chromosome.

To execute the program with a pre-trained model run:
```
$ python3 gnomix.py <query_file> <output_folder> <chr_nr> <phase> <path_to_model> 
```

where 
- <*query_file*> is a .vcf or .vcf.gz file containing the query haplotypes which are to be analyzed (see example in the **demo/data/** folder)
- <*output_folder*> is where the results will be written (see details in **Output** below and an example in the **demo/data/** folder)
- <*chr_nr*> is the chromosome number
- <*phase*> is either True or False corresponding to the intent of using the predicted ancestry for phasing correction (see details in **Phasing** below and in the **gnofix/** folder). Note that initial phasing (using a program like beagle, shapeit, or eagle) must still have been performed first.
- <*path_to_model*> is a path to the model used for predictions (see **Pre-trained Models** below)

### Downloading pre-trained models
In order to incorporate our pre-trained models into your pipeline, please use the following command to download pre-trained models for the whole human genome. The SNPs used for our pre-trained models are also included in the form of a plink .bim file for every chromosome.
```
sh download_pretrained_models.sh
```
This creates a folder called **pretrained_gnomix_models**. For each chromosome, we publish a *default_model.pkl* which can be used as a pre-trained model in the <*path_to_model*> field and a *.bim* file as explained above.

When making predictions, the input to the model is an intersection of the pre-trained model SNP positions and the SNP positions from the <query_file>. That means that the set of positions that are only in the original training input used to create the model (and not in the query samples) are encoded as missing, while the set of positions only in the <query_file> are discarded. We suggest that you attempt to have your query samples include as many model snps (listed in the .bim files) as possible for higher accuracy. When the script is executed, it will log the intersection-ratio between these model snps and the snps in your query samples, since the anceestry inference performance will depend on how many of the model's snp positions are missing in your query samples. If the intersection is low, we recommend training your own new model using references that contain all the snps in your query samples, or imputing your query samples to have the full set of snps present in the pre-trained model. N.B. Your query samples must have snps that are defined on the same strand as in the model. You can use the included model .bim files as a reference to find and then flip any snps in the query samples that are defined on the opposite strand. (If this step is not performed the query samples will appear to have snps containing variation unseen in the model's training and will thus be inferred with unexpected and unpredictable ancestries.)

The models named **default_model.pkl** are trained on hg build 37 references from the following biogeographic regions: *Subsaharan African (AFR), East Asian (EAS), European (EUR), Native American (NAT), Oceanian (OCE), South Asian (SAS), and West Asian (WAS)* and the model labels and predicts them as 0, 1, .., 6 respectively. The populations used to train these ancestries are given in the supplementary section of the reference provided at the bottom of this readme.

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

### Demo

After downloading our pre-trained models, one can demo the software in inference mode by running:
```
python3 gnomix.py demo/data/small_query_chr22.vcf.gz demo_output 22 True pretrained_gnomix_models/chr22/model_chm_22.pkl
```
This small query file contains only 9 samples of European, East Asian and African ancestry. The execution should take around a minute on a standard laptop. The inference can be analyzed, for example in the file demo_output/quer_results.msp, where we expect to see those three ancestries being inferred. For more details on those analysis, see the section on output below.

For more demos with training and larger datasets, see the [demo](demo.ipynb) notebook *demo.ipynb*.

### Advanced Options
More advanced configuration settings can be found in *config.yaml*. 
They include general settings, simulation settings and model settings. More details are given in the file itself. If training a model from scratch you can also pass an alternative config file as the last argument:

```
$ python3 gnomix.py <query_file> <output_folder> <chr_nr> <phase> <genetic_map_file> <reference_file> <sample_map_file> <config_file>
```

If no config is given, the program uses the default (*config.yaml*). The config file has advanced training options. Some of the parameters are
- verbose (bool) - verbosity
- simulation:
  - run: (bool) - whether to run simulation or not
  - path: (path) - if run is False, use data from this location. Must have been created by gnomix in the past.
  - rm_data (bool) - whether to remove simulated data (if memory constrained). It is set to false if run is False
  - r_admixed (float,positive) - number of simulated individuals generated = r_admixed x Size of sample map, default 1, set lower if memory is an issue. (To overcome memory constraints a minor allele frequency filter can also be used to remove very rare variants.)
  - splits: must contain train1, train2 and optionally validation. If validation ratio is 0, validation is not performed
  - generations indicates simulated individuals' generations since admixture. 
- model:
  - name (string) - model's name: default is "model"
  - inference (string) - 4 possible options - best / fast / large / default. "best" uses random string kernel base + xgboost smoother and is recommended for array data. "fast" uses logistic regression base + crf smoother. "large" uses logistic regression + convolutional smoother and is good for large datasets for which memory requirements are an issue. "default" uses logistic regression base + xgboost smoother and on whole genome has nearly the same accuracy as "best," but with much faster runtime.
  - window_size_cM (float, positive) -  size of window in centiMorgans
  - smooth_size (int, positive) - number of windows to be taken as context for smoother
  - context_ratio (float between 0 and 1) - context of base model windows
  - retrain_base (bool) - retrain base models with train2, validation data for a final base model
  - calibrate (bool) - if True, applies calibration on output probabilities
  - n_cores (int, positive) - how many units of cpu to use

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
To visualize the local ancestry output along the chromosome using [tagore](https://pypi.org/project/tagore/#usage) for plotting, see plot_chm from src/visualization.py and reference our [demo](demo.ipynb) example.

### Model
When training a model, the resulting model will be stored in *<output_folder>/models*. That way it can be re-used for analyzing another dataset.
The model's estimated accuracy is logged along with a confusion matrix which is stored in *<output_folder>/models/analysis*.

### Simulated data
The program simulates training data and stores it in *<output_folder>/generated_data*. To automatically remove the created data when training is done,
set *rm_simulated_data* to True in *config.yaml*. Note that in some cases, the simulated data can be re-used for training with similar settings. 
In those cases, not removing the data and then setting *run_simulation* to False will re-use the previously simulated data which can save a lot of time and compuation.

## Phasing

![Depiction of the process](src/Gnofix/figures/XGFix.gif)

Accurate phasing of genomic data is crucial for human demographic modeling and identity-by-descent analyses. It has been shown that leveraging information about an individual’s genomic ancestry improves performance of current phasing algorithms. Gnofix is a method that uses local ancestry inference to do exactly that. If you suspect your data might have phasing errors (generally the case unless trio phasing was possible), we recommend using this option <*phase*> as True. See the **gnofix/** folder for more details. 

![Local Ancestry for Phasing Error Correction](src/Gnofix/figures/laipec_resized.png)
Sequenced haplotypes phased with a phasing software (left). LAI is used to label haplotypes with ancestry predictions and phasing errors become evident (center). Phasing error correction using LAI is applied to correct phasing errors (right). Small numbers of phasing errors do not, however, impact the correct association of a variant with an ancestry, and so are typically only a visual nuisance.

## Calibration
To ensure that G-Nomix outputs probability estimates that reflect it's true confidence and accuracy, we recommend using calibration. We use Isotonic Regression to map the predicted probabilities to calibrated probabilities where the latter are more likely to have predictions with a confidence of X% correct matching their actual X% frequency of being correct in practice.

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as "academic research" must contact the [Stanford Office of Technology Licensing](https://otl.stanford.edu/) for a separate license. This applies to this repository directly and any other repository that includes source, executables, or git commands that pull/clone this repository as part of its function. Such repositories, whether ours or others, must include this notice. Academic users may fork this repository and modify and improve to suit their research needs, but also inherit these terms and must include a licensing notice to this effect.

## Cite

#### When using this software, please cite: 
### Helgi Hilmarsson, Arvind S Kumar, Richa Rastogi, Carlos D Bustamante, Daniel Mas Montserrat, Alexander G Ioannidis: "High Resolution Ancestry Deconvolution for Next Generation Genomic Data"

https://www.biorxiv.org/content/10.1101/2021.09.19.460980v1

```
@article {Hilmarsson2021.09.19.460980,
	author = {Hilmarsson, Helgi and Kumar, Arvind S and Rastogi, Richa and Bustamante, Carlos D and Mas Montserrat, Daniel and Ioannidis, Alexander G},
	title = {High Resolution Ancestry Deconvolution for Next Generation Genomic Data},
	elocation-id = {2021.09.19.460980},
	year = {2021},
	doi = {10.1101/2021.09.19.460980},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {As genome-wide association studies and genetic risk prediction models are extended to globally diverse and admixed cohorts, ancestry deconvolution has become an increasingly important tool. Also known as local ancestry inference (LAI), this technique identifies the ancestry of each region of an individual{\textquoteright}s genome, thus permitting downstream analyses to account for genetic effects that vary between ancestries. Since existing LAI methods were developed before the rise of massive, whole genome biobanks, they are computationally burdened by these large next generation datasets. Current LAI algorithms also fail to harness the potential of whole genome sequences, falling well short of the accuracy that such high variant densities can enable. Here we introduce G-Nomix, a set of algorithms that address each of these points, achieving higher accuracy and swifter computational performance than any existing LAI method, while also enabling portable models that are particularly useful when training data are not shareable due to privacy or other restrictions. We demonstrate G-Nomix (and its swift phase correction counterpart Gnofix) on worldwide whole-genome data from both humans and canids and utilize its high resolution accuracy to identify the location of ancient New World haplotypes in the Xoloitzcuintle, dating back over 100 generations. Code is available at https://github.com/AI-sandbox/gnomixCompeting Interest StatementCDB is the founder and CEO of Galatea Bio Inc and on the boards of Genomics PLC and Etalon.},
	URL = {https://www.biorxiv.org/content/early/2021/09/21/2021.09.19.460980},
	eprint = {https://www.biorxiv.org/content/early/2021/09/21/2021.09.19.460980.full.pdf},
	journal = {bioRxiv}
}

