"""
Define the core datatypes

"""
from dataclasses import dataclass
from ladataset import SimulationConfig
import numpy as np

class VCF:

    def __init__(self,vcf_file):
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

