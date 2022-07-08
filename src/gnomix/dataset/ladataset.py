from gnomix.core.datatypes import VCF, AncestryPredictions
import allel
import gzip
import pandas as pd
import os

from gnomix.core.utils import read_vcf, read_genetic_map

class LADataset:

    def __init__(self,
                 chm: int,
                 genetic_map_df: pd.DataFrame,
                 sample_map_df: pd.DataFrame,
                 ):

        self.chm = chm
        self.genetic_map_df = genetic_map_df


    # Initialize from VCF, sample map and genetic map
    @staticmethod
    def read_from_files(chm: str,
                        vcf: VCF, 
                        sample_map_file: str, 
                        genetic_map_file: str):

        assert os.path.exists(genetic_map_file)
        assert os.path.exists(sample_map_file)

        ## VCF data (vcf)
        chm = str(chm)
        snp_positions = vcf["variants/POS"].copy()
        ref_snps = vcf["variants/REF"].copy().astype(str)
        alt_snps = vcf["variants/ALT"][:,0].copy().astype(str)
        sample_names = []
        num_samples = len(sample_names)
        genotypes = None # Np array of size num_samples x num_snps x 2

        # Ancestry data (sample map)
        ancestries = None # Np array of size num_samples x num_snps x 2
        num_to_ancestry_map = {}

        # Genetic map data (genetic map)
        # provide a 1:1 map between positions : centimorgan
        # provide breakpoint proba for helping simulation
        snp_cms = None
        snp_bp = None

        return LADataset()

    def save_metadata(self):
        # Save everything other than genotypes and ancestries into a file
        metadata = {"snp_positions":self.snp_positions,
                    "ref_snps":self.ref_snps,
                    "alt_snps":self.alt_snps,
                    "num_to_ancestry_map":self.num_to_ancestry_map,
                    "chm":self.chm,
                    "genetic_map_data":self.genetic_map_data
                    }
        

    # Initialize from predictions
    @staticmethod
    def from_predictions(vcf_data: VCF,
                         predictions: AncestryPredictions):
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
    def simulate(self,sim_config):
        return LADataset()
