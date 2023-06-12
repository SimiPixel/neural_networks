import fnmatch
import os

import setuptools


def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)
    return list(paths)


setuptools.setup(
    name="neural_networks",
    packages=setuptools.find_packages(),
    version="0.4.14",
    package_data={
        "neural_networks": find_data_files("neural_networks", patterns=["*.joblib"])
    },
    include_package_data=True,
    install_requires=[
        "jax",
        "jaxlib",
        "joblib",
        "neptune",
        "x_xy @ git+https://github.com/SimiPixel/x_xy_v2.git",
        "dm-haiku",
        "tqdm",
    ],
)
