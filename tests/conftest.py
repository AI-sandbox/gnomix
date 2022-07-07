import pytest
import os

from tests.functional import paths

@pytest.fixture
def genetic_map_path(build: int = 37):

    if build == 37:
        return paths.GENETIC_MAP_FOLDER / "allchrs.b37.gmap"
    elif build == 38:
        print("We don't have that yet!!")
    else:
        print(f"Unknown build {build}")

@pytest.fixture
def temp_dir_path():
    temp_dir_path =  paths.TEST_DATA_FOLDER / "temp"
    if os.path.exists(temp_dir_path):
        print("Cleaning previous temp dirs...")
        os.system(f"rm -r {temp_dir_path}")
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    yield temp_dir_path
    print("Cleaning up tempt dir...")
    os.system(f"rm -r {temp_dir_path}")
