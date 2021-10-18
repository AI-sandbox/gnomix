# Configuration file templates

In this folder we have different conifiguration file tempaates corresponding to edge use cases. Some are critical for exectution, others only to ensure better results. If you find yourself tweeking a config file for your own special use case, consider sharing it (e.g. by creating an issue) so others researchers can also use it :)

## config.yaml

This is the default for all LAI on human genome and the same as the one we have in the root folder. It's designed for whole genome data but will work well with array data too.

## config_array.yaml

This is a slight improvement when using array data for the human genome. For array data, variations of the string kernels tend to be significantly superior to the default logistic regression model. Since array data is also smaller, the use of more complex models is not only very useful for performance also has less time and memory requirements. The only difference is the base models that are used: *model.inference=best*

## config_lowdata.yaml

When training with small samples size, e.g. less than 20 samples or less then 5 samples per population, one can barely afford data for validation. That's why we recommend having none: *simulation.split.ratios.val = 0*. Given the small data size, we also recommend using the best models independently from the data type, *model.inference=best*. If compuation time is to high, the default (empty) is better.

## config_plants.yaml

This config has been used for analyzing date palm data. It's an extension of the *lowdata* config file with the additional change in the window size. Since that specific data had much higher density of snps per centiMorgan we reduce the window size of the model:

- *window_size_cM = 0.05* (smaller)

- *smooth_size = 25* (smaller)

## Coming soon: templatets for dogs

