from ladataset import LADataset
from datatypes import VCF, BaseProbabilities

class BaseModels:

    def __init__(self, type, parallel_training=False):
        pass

    def train(ladataset: LADataset):
        return

    def infer(vcf: VCF) -> BaseProbabilities:
        return

    
