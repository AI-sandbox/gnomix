# General configuration
verbose = True     # logging
instance_name = "" # can be used for organizing simulation output and models

# Simulatiation configuration
run_simulation = False    # if you already simulated data, setting False will re-use the data
founders_ratios = [0.8, 0.15, 0.05]
# num_outs = [900, 180, 90] # how many individuals to simulate in each set
r_admixed = 3.0 # number of individuals as a function of founders_ratios.
generations = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48] # generations to generate during simulation
rm_simulated_data = False     # remove the simulated data after training the model

# Model configuration
model_name = "model" # the complete name will be <model_name>_chm<chm>.pkl
window_size_cM = 0.20# window size of the XGMix model in centiMorgans
smooth_size = 75     # how many windows to aggregate over in second phase
context_ratio = 0.0
missing = 0.0        # fraction in [0,1) of how much to simulate missing data during training
retrain_base = True  # for retraining base models with [train1, train2] once the smoother has been trained with train2
calibrate = True
n_cores = 30         # how many units of cpu to use
mode_filter_size = 5
smooth_depth = 4