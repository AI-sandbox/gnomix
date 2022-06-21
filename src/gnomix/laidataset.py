import numpy as np
import pandas as pd
import os
from collections import namedtuple
import scipy.interpolate

from gnomix.utils import read_vcf, read_genetic_map

def get_chm_info(genetic_map,variants_pos,chm):

    """
    get chromosome length in morgans from genetic map.
    Assumes genetic_map is sorted.
    
    genetic_map: file with the following format
    variants: a npy array with numbers representing centi morgans
    
    """
    genetic_chm = read_genetic_map(genetic_map_path=genetic_map, chm=chm)

    # get length of chm.
    chm_length_morgans = max(genetic_chm["pos_cm"])/100.0

    # get snp info - snps in the vcf file and their cm values.
    # then compute per position probability of being a breapoint.
    # requires some interpolation and finding closest positions.
    """
    # 1: Minimum in a sorted array approach and implemented inside admix().
        - O(logn) every call to admix. Note that admix() takes O(n) anyway.
    # 2: Find probabilities using span. - One time computation.

    """
    # This adds 0 overhead to code runtime.
    # get interpolated values of all reference snp positions
    genomic_intervals = scipy.interpolate.interp1d(x=genetic_chm["pos"].to_numpy(), y=genetic_chm["pos_cm"].to_numpy(),fill_value="extrapolate")
    genomic_intervals = genomic_intervals(variants_pos)

    # interpolation
    lengths = genomic_intervals[1:] - genomic_intervals[0:-1]
    bp = lengths / lengths.sum()

    return chm_length_morgans, bp

def get_sample_map_data(sample_map, sample_weights=None):
    sample_map_data = pd.read_csv(sample_map,delimiter="\t",header=None,comment="#", dtype="object")
    sample_map_data.columns = ["sample","population"]

    # creating ancestry map into integers from strings
    # id is based on order in sample_map file.
    ancestry_map = {}
    curr = 0
    for i in sample_map_data["population"]:
        if i in ancestry_map.keys():
            continue
        else:
            ancestry_map[i] = curr
            curr += 1
    sample_map_data["population_code"] = sample_map_data["population"].apply(ancestry_map.get)

    if sample_weights is not None:
        sample_weights_df = pd.read_csv(sample_weights,delimiter="\t",header=None,comment="#")
        sample_weights_df.columns = ["sample","sample_weight"]
        sample_map_data = pd.merge(sample_map_data, sample_weights_df, on='sample')

    else:
        sample_map_data["sample_weight"] = [1.0/len(sample_map_data)]*len(sample_map_data)

    return ancestry_map, sample_map_data


Person = namedtuple('Person', 'maternal paternal name')

def build_founders(sample_map_data,gt_data,chm_length_snps):
    
    """
    Returns founders - a list of Person datatype.
    founders_weight - a list with a weight for each sample in founders

    Inputs
    gt_data shape: (num_snps, num_samples, 2)
    
    """

    # building founders
    founders = []

    for i in sample_map_data.iterrows():

        # first get the index of this sample in the vcf_data.
        # if not there, skip and print to log.

        index = i[1]["index_in_reference"]

        name = i[1]["sample"]

        # when creating maternal, paternal make sure it has same keys

        maternal = {}
        paternal = {}

        # let us use the first for maternal in the vcf file...
        maternal["snps"] = gt_data[:,index,0].astype(np.uint8)
        paternal["snps"] = gt_data[:,index,1].astype(np.uint8)

        # single ancestry assumption.
        maternal["anc"] = np.array([i[1]["population_code"]]*chm_length_snps).astype(np.uint8)
        paternal["anc"] = np.array([i[1]["population_code"]]*chm_length_snps).astype(np.uint8)

        # any more info like coordinates, prs can be added here.

        p = Person(maternal,paternal,name)

        founders.append(p)
    
    return founders



def admix(founders,founders_weight,gen,breakpoint_probability,chm_length_snps,chm_length_morgans):

    """
    create an admixed haploid from the paternal and maternal sequences
    in a non-recursive way.
    
    Returns:
    haploid_returns: dict with same keys as self.maternal and self.paternal

    """
    # assert all founders have all keys.

    assert len(founders) >= 2, "Too few founders!!!"
    order = sorted(founders[0].maternal.keys())
    
    # for each gen, we sample from poisson
    num_crossovers = int(sum(np.random.poisson(chm_length_morgans,size=gen)))

    # initilizing all numbers to 0.
    haploid_returns = {}
    for key in order:
        haploid_returns[key] = np.zeros_like(founders[0].maternal[key])
    
    # edge case of no breaking points.
    if num_crossovers == 0:
            
        haploid_returns = {}
        select_id = np.random.choice(len(founders),p=founders_weight)
        select = founders[select_id]
        choice = np.random.rand()>=0.5
        select = select.maternal if choice else select.paternal
        for key in order:
            
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
            for key in order:
                haploid_returns[key][begin:end] = select[key][begin:end].copy()

    return haploid_returns



def write_output(root,dataset):
    
    # dataset is a list of Person
    
    if not os.path.isdir(root):
        os.makedirs(root)
        
    snps = []
    anc = []
    for person in dataset:
        snps.append(person.maternal["snps"])
        snps.append(person.paternal["snps"])
        anc.append(person.maternal["anc"])
        anc.append(person.paternal["anc"])

    # create npy files.
    snps = np.stack(snps)
    np.save(root+"/mat_vcf_2d.npy",snps)

    # create map files.
    anc = np.stack(anc)
    np.save(root+"/mat_map.npy",anc)


class LAIDataset:
    
    
    def __init__(self,chm,reference,genetic_map,seed=94305):

        np.random.seed(seed)

        self.chm = chm
        
        # vcf data
        print("Reading vcf file...")
        vcf_data = read_vcf(reference,self.chm)
        self.pos_snps = vcf_data["variants/POS"].copy()
        self.num_snps = vcf_data["calldata/GT"].shape[0]
        self.ref_snps = vcf_data["variants/REF"].copy().astype(str)
        self.alt_snps = vcf_data["variants/ALT"][:,0].copy().astype(str)
        
        self.call_data = vcf_data["calldata/GT"]
        self.vcf_samples = vcf_data["samples"]

        # genetic map data
        print("Getting genetic map info...")
        self.morgans, self.breakpoint_prob = get_chm_info(genetic_map, self.pos_snps, self.chm)
        
    
    def buildDataset(self, sample_map, sample_weights=None):
        
        """
        reads in the above files and extacts info
        
        self: chm, num_snps, morgans, breakpoint_prob, splits, pop_to_num, num_to_pop
        sample_map_data => sample name, population, population code, (maternal, paternal, name), weight, split
        """
        
        # sample map data
        print("Getting sample map info...")
        self.pop_to_num, self.sample_map_data = get_sample_map_data(sample_map, sample_weights)
        self.num_to_pop = {v: k for k, v in self.pop_to_num.items()}
        
        try:
            map_samples = np.array(list(self.sample_map_data["sample"]))

            sorter = np.argsort(self.vcf_samples)
            indices = sorter[np.searchsorted(self.vcf_samples, map_samples, sorter=sorter)]
            self.sample_map_data["index_in_reference"] = indices
            
        except:
            raise Exception("sample not found in vcf file!!!")
        
        # self.founders
        print("Building founders...")
        self.sample_map_data["founders"] = build_founders(self.sample_map_data,self.call_data,self.num_snps)
        self.sample_map_data.drop(['index_in_reference'], axis=1, inplace=True)
        
    def __len__(self):
        return len(self.sample_map_data)
    
    def data(self):
        return self.sample_map_data
    
    def metadata(self):
        metadict = {
            "chm":self.chm,
            "morgans":self.morgans,
            "num_snps":self.num_snps,
            "pos_snps":self.pos_snps,
            "ref_snps":self.ref_snps,
            "alt_snps":self.alt_snps,
            "pop_to_num":self.pop_to_num,
            "num_to_pop":self.num_to_pop
        }
        return metadict
        
    def split_sample_map(self, ratios, split_names=None):
        """
        Given sample_ids, populations and the amount of data to be put into each set,
        Split it such that all sets get even distribution of sample_ids for each population.
        """

        assert sum(ratios) == 1, "ratios must sum to 1"

        split_names = ["set_"+str(i) for i in range(len(ratios))] if split_names is None else split_names
        
        set_ids = [[] for _ in ratios]
        
        for p in np.unique(self.sample_map_data["population"]):

            # subselect population
            pop_idx = self.sample_map_data["population"] == p
            pop_sample_ids = list(np.copy(self.sample_map_data["sample"][pop_idx]))
            n_pop = len(pop_sample_ids)

            # find number of samples in each set
            n_sets = [int(round(r*n_pop)) for r in ratios]
            while sum(n_sets) > n_pop:
                n_sets[0] -= 1    
            while sum(n_sets) < n_pop:
                n_sets[-1] += 1

            # divide the samples accordingly
            for s, r in enumerate(ratios):
                n_set = n_sets[s]
                set_ids_idx = np.random.choice(len(pop_sample_ids),n_set,replace=False)
                set_ids[s] += [[pop_sample_ids.pop(idx), p, split_names[s]] for idx in sorted(set_ids_idx,reverse=True)]

        split_df = pd.DataFrame(np.concatenate(set_ids), columns=["sample", "population", "split"])
        return split_df

    def include_all(self, from_split, in_split):
        from_split_data = self.sample_map_data[self.sample_map_data["split"]==from_split]
        from_pop = np.unique(from_split_data["population"])
        ave_pop_size = np.round(len(from_split_data)/len(from_pop))

        in_split_data = self.sample_map_data[self.sample_map_data["split"]==in_split]
        in_pop = np.unique(in_split_data["population"])

        missing_pops = [p for p in from_pop if p not in in_pop]

        if len(missing_pops) > 0:
            print("WARNING: Small sample size from populations: {}".format(np.array(missing_pops)))
            print("... Proceeding by including duplicates in both base- and smoother data...")
            for p in missing_pops:
                # add some amount of founders to in_pop
                from_founders = from_split_data[from_split_data["population"] == p].copy()
                n_copies = min(ave_pop_size, len(from_founders))
                copies = from_founders.sample(n_copies)
                copies["split"] = [in_split]*n_copies
                self.sample_map_data = self.sample_map_data.append(copies)

    def create_splits(self,splits,outdir=None):
        print("Splitting sample map...")
        
        # splits is a dict with some proportions, splits keys must be str
        assert(type(splits)==dict)
        self.splits = splits
        split_names, prop = zip(*self.splits.items())

        # normalize
        prop = np.array(prop) / np.sum(prop)
    
        # split founders randomly within each ancestry
        split_df = self.split_sample_map(ratios=prop, split_names=split_names)
        self.sample_map_data = self.sample_map_data.merge(split_df, on=["sample", "population"])
        self.include_all(from_split="train1",in_split="train2")

        # write a sample map to outdir/split.map
        if outdir is not None:
            for split in splits:
                split_file = os.path.join(outdir,split+".map")
                self.return_split(split)[["sample","population"]].to_csv(split_file,sep="\t",header=False,index=False)
        
    def return_split(self,split):
        if split in self.splits:
            return self.sample_map_data[self.sample_map_data["split"]==split]
        else:
            raise Exception("split does not exist!!!")
        
        
    def simulate(self,num_samples,split="None",gen=None,outdir=None,return_out=True, verbose=False):
        
        # general purpose simulator: can simulate any generations, either n of gen g or 
        # just random n samples from gen 2 to 100.
        
        assert(type(split)==str)
        if verbose:
            print("Simulating using split: ",split) 
        
        # get generations for each sample to be simulated
        if gen == None:
            gens = np.random.randint(2,100,num_samples)
            if verbose:
                print("Simulating random generations...")
            
        else:
            gens = gen * np.ones((num_samples),dtype=int)
            if verbose:
                print("Simulating generation: ",gen)
        
        # corner case
        if gen == 0:
            simulated_samples =  self.sample_map_data[self.sample_map_data["split"]==split]["founders"].tolist()
            if outdir is not None:
                if verbose:
                    print("Writing simulation output to: ",outdir)
                write_output(outdir,simulated_samples)
        
            # return the samples
            if return_out:
                return simulated_samples
            else:
                return
        
        # get the exact founder data based on split
        founders = self.sample_map_data[self.sample_map_data["split"]==split]["founders"].tolist()
        founders_weight = self.sample_map_data[self.sample_map_data["split"]==split]["sample_weight"].to_numpy()
        founders_weight = list(founders_weight/founders_weight.sum()) # renormalize to 1
        if len(founders) == 0:
            raise Exception("Split does not exist!!!")
        
        # run simulation
        if verbose:
            print("Generating {} admixed samples".format(num_samples))
        simulated_samples = []
        for i in range(num_samples):
            
            # create an admixed Person
            maternal = admix(founders,founders_weight,gens[i],self.breakpoint_prob,self.num_snps,self.morgans)
            paternal = admix(founders,founders_weight,gens[i],self.breakpoint_prob,self.num_snps,self.morgans)
            name = "admixed"+str(int(np.random.rand()*1e6))
            
            adm = Person(maternal,paternal,name)
            simulated_samples.append(adm)
            
        # write outputs
        if outdir is not None:
            if verbose:
                print("Writing simulation output to: ",outdir)
            write_output(outdir,simulated_samples)
            # TODO: optionally, we can even convert these to vcf and result (ancestry) files
        
        # return the samples
        if return_out:
            return simulated_samples
        else:
            return
