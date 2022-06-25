from pathlib import Path
import os
from gnomix.paths import TEST_CONFIG_PATH
import numpy as np

from gnomix.utils import (
    get_window_level_local_ancestry_inference_from_msp_file,
    get_labels_from_sample_map_file
)
from gnomix.generate_dataset_from_1kg import generate_dataset

def test_train_and_inference_single_ancestry_small(
    genetic_map_path: Path,
    temp_dir_path: Path
) -> None:

    chromosome = "22"
    should_phase = "False"

    reference_file, reference_sample_map_path, query_file, query_sample_map_path = generate_dataset(name="small_test")

    # defining and executing the command
    run_cmd =  "python3 run_gnomix.py"
    train_cmd = " ".join([
        run_cmd,
        str(query_file),
        str(temp_dir_path),
        chromosome,
        should_phase,
        str(genetic_map_path),
        str(reference_file),
        str(reference_sample_map_path),
        str(TEST_CONFIG_PATH)
    ])
    print("Running from command line: \n\t", train_cmd)
    cmd_status = os.system(train_cmd)
    assert cmd_status == 0, "Execution not successful"

    # evaluating
    inferred_ancestry_labels = get_window_level_local_ancestry_inference_from_msp_file(
        path_to_msp_file=temp_dir_path / "query_results.msp"
    )
    true_gloabl_ancestry_labels = get_labels_from_sample_map_file(
        sample_map_file=query_sample_map_path,
        dtype=inferred_ancestry_labels.dtype
    )
    accuracy = np.mean(inferred_ancestry_labels == true_gloabl_ancestry_labels)
    print(f"Accuracy: {accuracy*100:.2f}%")
    assert accuracy > 0.95


