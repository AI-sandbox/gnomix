import pathlib
import pkg_resources

DATA_FOLDER = pathlib.Path(pkg_resources.resource_filename('gnomix', 'data/'))
TEST_DATA_FOLDER = DATA_FOLDER / "test"
DEMO_DATA_FOLDER = DATA_FOLDER / "demo"
GENETIC_MAP_FOLDER = DATA_FOLDER / "genetic_maps"

ONE_KG_DATA_FOLDER = DATA_FOLDER / "1000genomes"
SEQUENCE_DATA_FILE_1KG = ONE_KG_DATA_FOLDER / f"ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
SAMPLE_MAP_FILE_1KG = ONE_KG_DATA_FOLDER / "1000g.smap"

CONFIG_PATH = pathlib.Path(pkg_resources.resource_filename('gnomix', 'configs'))
DEFAULT_CONFIG_PATH = CONFIG_PATH / "default.yaml"
TEST_CONFIG_PATH = CONFIG_PATH / "test.yaml"