#!/bin/bash

# Usage:
#  $ NVCR_PAT=YOUR_TOKEN ./docker/ci/build_and_push.sh

set -e

IMAGE_NAME=nvcr.io/nvidian/crossfit-ci
IMAGE_TAG=23.10

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f docker/ci/Dockerfile .

#if [ -z ${NVCR_PAT+x} ]
#then
#  echo "This script assumes your nvcr personal access token is stored in "
#  echo "a variable named NVCR_PAT, e.g., export NVCR_PAT=YOUR_TOKEN. "
#  exit 1
#fi
#
#echo ${NVCR_PAT} | docker login nvcr.io -u '$oauthtoken' --password-stdin
#
#docker push ${IMAGE_NAME}:${IMAGE_TAG}
