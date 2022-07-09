from curses import meta
from dataclasses import dataclass
from gnomix.core.datatypes import VCF, AncestryPredictions
import allel
import gzip
import pandas as pd
import os

from gnomix.core.utils import read_vcf, read_genetic_map
from typing import List, TypedDict
from pathlib import Path
from collections import namedtuple
from gnomix.dataset.laidataset import LAIDataset
import numpy as np

class Haplotype:

    def __init__(self,
                 snps: List[bool], 
                 anc: List[int]):
        self.snps =  snps
        self.anc = anc
    
    def __getitem__(self,
                 key: str):
        if key == "anc":
            return self.anc

        elif key == "snps":
            return self.snps

        else:
            return None

ORDER = ["anc","snps"]

@dataclass
class Person:
    name: str
    maternal: Haplotype
    paternal: Haplotype

def admix(founders: List[Person],
          gen: int,
          breakpoint_probability: List[float],
          chm_length_snps: int,
          chm_length_morgans: float) -> Haplotype:

    """
    create an admixed haploid from the paternal and maternal sequences
    in a non-recursive way.
    
    Returns:
    haploid_returns: dict with same keys as self.maternal and self.paternal

    """
    # assert all founders have all keys.

    assert len(founders) >= 2, "Too few founders!!!"

    # Assume equal weight for each founder
    founders_weight = np.ones(len(founders)) / len(founders)
    
    # for each gen, we sample from poisson
    num_crossovers = int(sum(np.random.poisson(chm_length_morgans,size=gen)))

    # initilizing all numbers to 0.
    haploid_returns = {}
    for key in ORDER:
        haploid_returns[key] = np.zeros_like(founders[0].maternal[key])
    
    # edge case of no breaking points.
    if num_crossovers == 0:
            
        haploid_returns = {}
        select_id = np.random.choice(len(founders),p=founders_weight)
        select = founders[select_id]
        choice = np.random.rand()>=0.5
        select = select.maternal if choice else select.paternal
        for key in ORDER:
            
            haploid_returns[key] = select[key].copy()

    else:
        
        breakpoints = np.random.choice(np.arange(1,chm_length_snps), 
                                        size=num_crossovers, 
                                        replace=False, 
                                        p=breakpoint_probability)
        breakpoints = np.sort(breakpoints)
        
        breakpoints = np.concatenate(([0],breakpoints,[chm_length_snps]))
        
        # select paternal or maternal randomly and apply crossovers.
        for i in range(len(breakpoints)-1):
            begin = breakpoints[i]
            end = breakpoints[i+1]
            # choose random founder for this segment, then choose random haplotype for this founder
            select_id = np.random.choice(len(founders),p=founders_weight)
            select = founders[select_id]
            choice = np.random.rand()>=0.5
            select = select.maternal if choice else select.paternal
            for key in ORDER:
                haploid_returns[key][begin:end] = select[key][begin:end].copy()

    return Haplotype(haploid_returns)


class LADataset:

    def __init__(self,
                 chm: int,
                 snp_positions: List[int],
                 ref_snps: List[str],
                 alt_snps: List[str]
                 ):

        self.chm = chm
        self.snp_positions = snp_positions
        self.ref_snps = ref_snps
        self.alt_snps = alt_snps

        self.num_snps = len(self.snp_positions)

        self.samples = []

        self.genetic_info_configured = False
        self.anc_configured = False

    def configure_genetic_map_info(self,
                        genetic_map_file: Path):
    
        if self.genetic_info_configured:
            print("Cannot configure it! already done")
            return

        self.snp_cm = []
        self.snp_bp = []
        self.morgans = None

        self.genetic_info_configured = True
        return

    def configure_genetic_map_info(self,
                             snp_cm: List[float],
                             snp_bp: List[float],
                             morgans: float):


        if self.genetic_info_configured:
            print("Cannot configure it! already done")
            return

        self.snp_cm = snp_cm
        self.snp_bp = snp_bp
        self.morgans = morgans

        self.genetic_info_configured = True
        return

    def configure_ancestry(self,
                           num_to_pop: dict):

        if self.anc_configured:
            print("Cannot configure it! already done")
            return

        self.num_to_pop = num_to_pop

        self.anc_configured = True
        return

    def add_sample(self,
                   sample_name: str,
                   m_gt: List[bool],
                   p_gt: List[bool],
                   ancestry: int):
        
        if not self.anc_configured or not self.genetic_info_configured:
            print("Failed to add founder. Please add genetic map info and ancestry map info")
            return 

        assert len(m_gt) == len(p_gt), "Maternal and paternal need to be equal"

        maternal_haplo = Haplotype(m_gt, [ancestry]*len(m_gt))
        paternal_haplo = Haplotype(p_gt, [ancestry]*len(p_gt))
        person = Person(sample_name,maternal_haplo,paternal_haplo)
        self.samples.append(person)

    def add_sample(self,
                   sample_name: str,
                   m_gt: List[bool],
                   p_gt: List[bool],
                   m_anc: List[int],
                   p_anc: List[int]):

        if not self.anc_configured or not self.genetic_info_configured:
            print("Failed to add founder. Please add genetic map info and ancestry map info")

        assert len(m_gt) == len(p_gt), "Maternal and paternal need to be equal"
        assert len(m_anc) == len(p_anc), "Maternal and paternal need to be equal"
        assert len(m_gt) == len(m_anc), "snps and anc need to be equal"

        maternal_haplo = Haplotype(m_gt, m_anc)
        paternal_haplo = Haplotype(p_gt, p_anc)
        person = Person(sample_name,maternal_haplo,paternal_haplo)
        self.samples.append(person)
        
    def metadata(self):
        # Save everything other than genotypes and ancestries into a file
        metadata = {"snp_positions":self.snp_positions,
                    "ref_snps":self.ref_snps,
                    "alt_snps":self.alt_snps,
                    "num_to_ancestry_map":self.num_to_pop,
                    "chm":self.chm,
                    "genetic_map_data":self.genetic_map_data
                    }
        return metadata

    def replica(self):

        replica = LAIDataset(self.chm, self.snp_pos, self.ref_snps, self.alt_snps)
        replica.configure_genetic_map_info(self.snp_cm, self.snp_bp, self.morgans)
        replica.configure_ancestry(self.num_to_pop)

        return replica


    def visualize(self,num_samples=None):
        ### Plot all self.samples / num_samples of self.samples
        return

    ## Simulation
    def simulate(self,num_samples,gens,verbose=True) -> LAIDataset:
        assert len(gens) == num_samples, "gens must be list with each element being generation of a sample"

        gens = gen * np.ones((num_samples),dtype=int)
        if verbose:
            print("Simulating")

        if 0 in gens:
            print("Cannot simulate generation 0")
            return
        
        simulated_samples = []
        for i in range(num_samples):
            gen = gens[i]

            # create an admixed Person
            maternal = admix(self.samples,gen,self.snps_bp,self.num_snps,self.morgans)
            paternal = admix(self.samples,gen,self.snps_bp,self.num_snps,self.morgans)
            name = "admixed"+str(int(np.random.rand()*1e6))
            
            adm = Person(name,maternal,paternal)
            simulated_samples.append(adm)

        laid = self.replica()
        laid.samples = simulated_samples

        return laid

    def merge(self,other):
        ## assert self and other have same metadata fields
        assert self.metadata() == other.metadata(), "Cannot be merged"
        self.samples += other.samples


# Helper function to conver laidataset to numpy
# In the future the internal representation will change to something more efficient
# This needs to be implemented for the particular format by the caller function
def ladataset_to_numpy(ladataset):
    snps = []
    anc = []
    for person in ladataset.samples:
        snps.append(person.maternal["snps"])
        snps.append(person.paternal["snps"])
        anc.append(person.maternal["anc"])
        anc.append(person.paternal["anc"])

    # create npy files.
    snps = np.stack(snps)

    # create map files.
    anc = np.stack(anc)

    return snps, anc

