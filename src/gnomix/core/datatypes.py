"""
Define the core datatypes

"""
from dataclasses import dataclass
from gnomix.dataset.ladataset import SimulationConfig
import numpy as np

class VCF:

    def __init__(self,vcf_file):
        pass

    def checks(self):
        # Make sure phased

        # Warning for missing values

        # Make sure filter passed for all

        # Keep the INFO fields

        # Check if only bi-allelic

        pass

    def save_metadata(self):
        """
        Save everything other than genotype data into a file
        """
        pass



BaseProbabilities = np.ndarray

AncestryPredictions = np.ndarray

class GM:

    def __init__(self,basemodels,smoother):
        pass

    @staticmethod
    def from_file(file):
        return GM()


EvalSummary = dict

@dataclass
class SimulationConfig:

    generations: list
    run: bool
    path: str
    r_admixed: float = 2.0
    remove: bool = False
    val_split: float = 0.05

@dataclass
class ExperimentConfig:

    simulation_config: SimulationConfig

## Folders and file types

def validate_simulation_dir(path):
    return True


def validate_model_dir(path):
    return True

def validate_lai_file(path):
    return True

