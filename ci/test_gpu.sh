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
  "pytorch>=2.0,<=*cuda*"
  transformers \
  pytest \
  sentence-transformers  \
  sentencepiece \
  -c conda-forge \
  -c nvidia \
  --yes

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
