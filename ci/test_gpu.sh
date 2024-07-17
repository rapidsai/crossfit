#!/bin/bash
# Enabling strict error handling
set -Eeuo pipefail

echo "Checking CUDA version in the conda environment..."

# Extract CUDA version from conda list output
CUDA_VERSION=$(conda list | grep 'cuda-version' | awk '{print $2}')

# Check if CUDA version was found
if [ -z "$CUDA_VERSION" ]; then
    echo "CUDA version not found in the conda environment."
    exit 1  # Exit with a non-zero status indicating failure
else
    echo "CUDA version found: $CUDA_VERSION"
fi

echo "Installing pytorch,transformers and pytest to the environment for crossfit tests..."
mamba install \
  cuda-version=$CUDA_VERSION \
  conda-forge::pytorch \
  conda-forge::transformers \
  conda-forge::pytest \
  -c conda-forge \
  --override-channels \
  --yes

# Have to install sentence-transformers from pip
# because conda-forge leads to a torch vision conflict
# which leads to it being installed on CPUs
pip3 install sentence-transformers sentencepiece

# Install the crossfit package in editable mode with test dependencies
pip3 install -e '.[test]'
# Running tests
echo "Running tests..."
pytest tests
# Capture the exit code of pytest
EXITCODE=$?

# Echo the exit code
echo "Crossfit test script exiting with value: ${EXITCODE}"

# Exit with the same code as pytest
exit ${EXITCODE}
