import numpy as np
import os
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

from gnomix.paths import (
    DATA_FOLDER,
    TEST_DATA_FOLDER,
    DEMO_DATA_FOLDER,
    SEQUENCE_DATA_FILE_1KG,
    SAMPLE_MAP_FILE_1KG
)

CHR = "22"

SEED = 94305

FILENAMES = ["reference.vcf", "reference.smap", "query.vcf", "query.smap"]

def parse_args():
    parser = argparse.ArgumentParser("Dataset generation")
    parser.add_argument('--name', choices=['demo', 'small_test'], help='Dataset name')
    args = parser.parse_args()
    return args


def subset_from_sequence_data(sample_map, sample_name, root_folder, sequencing_data_file):

    print(f"Using bcftools to create vcf with the {sample_name} samples...")

    sample_file = root_folder / f"{sample_name}_samples.tsv"
    vcf_file = root_folder / f"{sample_name}.vcf"
    sample_map_path = root_folder / f"{sample_name}.smap"
    sample_map.to_csv(sample_map_path, sep="\t", index=False)

    np.savetxt(sample_file, sample_map["#Sample"].tolist(), delimiter="\t", fmt="%s")

    subset_cmd = "bcftools view" +\
        f" -S {str(sample_file)}" +\
        f" -o {str(vcf_file)}" +\
        f" {sequencing_data_file}"
    print("Running in command line: \n\t", subset_cmd)
    os.system(subset_cmd)

    os.system(f"rm {sample_file}")

    return vcf_file, sample_map_path

def get_demo_split(sample_map):

    reference_sample_map, query_sample_map = train_test_split(
        sample_map,
        test_size=0.2,
        random_state=SEED
    )

    return DEMO_DATA_FOLDER, reference_sample_map, query_sample_map

def get_small_test_split(sample_map):

    dataset_folder = TEST_DATA_FOLDER / "small"

    dataset_sample_map = sample_map.sample(frac=0.2)

    reference_sample_map, query_sample_map = train_test_split(
        dataset_sample_map,
        test_size=0.2,
        random_state=SEED
    )

    return dataset_folder, reference_sample_map, query_sample_map

def generate_dataset(name):

    assert os.path.exists(DATA_FOLDER), f"No data found at {DATA_FOLDER}."

    sample_map = pd.read_csv(SAMPLE_MAP_FILE_1KG, sep="\t")
    
    if name == "small_test":
        dataset_folder, reference_sample_map, query_sample_map = get_small_test_split(sample_map)
    elif name == "demo":
        dataset_folder, reference_sample_map, query_sample_map = get_demo_split(sample_map)

    dataset_paths = [dataset_folder / filename for filename in FILENAMES]
    dataset_files_exist = [os.path.exists(path) for path in dataset_paths]
    if all(dataset_files_exist):
        print(f"data already exists at {dataset_folder}! skipping...")
    else:
        print(f"Creating dataset for {dataset_folder}")

        dataset_folder.mkdir(exist_ok=True)

        subset_from_sequence_data(
            sample_map=reference_sample_map, 
            sample_name="reference",
            root_folder=dataset_folder,
            sequencing_data_file=SEQUENCE_DATA_FILE_1KG
        )

        subset_from_sequence_data(
            sample_map=query_sample_map, 
            sample_name="query",
            root_folder=dataset_folder,
            sequencing_data_file=SEQUENCE_DATA_FILE_1KG
        )

    return dataset_paths

if __name__ == "__main__":

    args = parse_args()
    generate_dataset(name=args.name)


 