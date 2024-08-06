# crossfit

[![PyPI](https://img.shields.io/pypi/v/crossfit.svg)](https://pypi.org/project/crossfit/)
[![Changelog](https://img.shields.io/github/v/release/rapidsai/crossfit?include_prereleases&label=changelog)](https://github.com/rapidsai/crossfit/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/rapidsai/crossfit/blob/main/LICENSE)

Multi Node Multi GPU Offline Inference and metric calculation library

## Installation

Install this library using `pip`:

    pip install crossfit

### Installation from source (for cuda 12.x)

```
git clone https://github.com/rapidsai/crossfit.git
cd crossfit
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
```

## Usage

Usage instructions go here.

## Development

To contribute to this library, first create a conda environment with the necessary dependencies:
```
cd crossfit
mamba env create -f conda/environments/cuda_dev.yaml
conda activate crossfit_dev
```

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
