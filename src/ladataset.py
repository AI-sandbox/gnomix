from datatypes import VCF, AncestryPredictions
import allel
import gzip
import pandas as pd
import os

from src.gnomix.utils import read_vcf, read_genetic_map

class LADataset:

    def __init__(self,
                 vcf: VCF, 
                 genetic_map_file: str,
                 sample_map_file: str,
                 chm: int):

        assert os.path.exists(genetic_map_file)
        assert os.path.exists(sample_map_file)

        self.chm = str(chm)
        
        self.genotypes = []
        self.ancestries = []


        ## Sequencing metadata
        self.pos_snps = vcf["variants/POS"].copy()
        self.num_snps = vcf["calldata/GT"].shape[0]
        self.ref_snps = vcf["variants/REF"].copy().astype(str)
        self.alt_snps = vcf["variants/ALT"][:,0].copy().astype(str)
        
        # self.call_data = vcf["calldata/GT"]
        self.phased = True

    # Initialize from VCF, sample map and genetic map
    @staticmethod
    def read_from_files(vcf: VCF, sample_map_file: str, genetic_map_file: str):
        return LADataset()

    # Initialize from predictions
    @staticmethod
    def from_predictions(vcf_data: VCF,
                         predictions: AncestryPredictions):
        return LADataset()

    
    # Initialize from simulation
    def simulated_ladataset(self,simulated_samples,simulated_ancestries):
        return LADataset()

    # Writers for various formats
    def write_vcf(self):
        pass

    def write_lai(self):
        pass

    def write_fb(self):
        pass

    def write_msp(self):
        pass

    ## Visualizer functions
    def visualize_sample(self,sample_id):
        return

    def visualize_dataset(self):
        return

    ## Simulation
    def simulate(self,sim_config) -> LADataset:
        return
