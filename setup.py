# Copyright 2024 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import itertools
import os

from setuptools import find_packages, setup

VERSION = "0.0.7"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


def read_requirements(filename):
    base = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(base, filename), "r", "utf-8") as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


_dev = read_requirements("requirements/dev.txt")

requirements = {
    "base": read_requirements("requirements/base.txt"),
    "cuda12x": read_requirements("requirements/cuda12x.txt"),
    "dev": _dev,
    "tensorflow": read_requirements("requirements/tensorflow.txt"),
    "pytorch": read_requirements("requirements/pytorch.txt"),
    "jax": read_requirements("requirements/jax.txt"),
}

dev_requirements = {
    "cuda12x-dev": requirements["cuda12x"] + _dev,
    "tensorflow-dev": requirements["tensorflow"] + _dev,
    "pytorch-dev": requirements["pytorch"] + _dev,
    "jax-dev": requirements["jax"] + _dev,
}


setup(
    name="crossfit",
    description="Offline inference and metric calculation library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    url="https://github.com/rapidsai/crossfit/",
    project_urls={
        "Issues": "https://github.com/rapidsai/crossfit/issues",
        "CI": "https://github.com/rapidsai/crossfit/actions/",
        "Changelog": "https://github.com/rapidsai/crossfit/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=find_packages(),
    package_dir={"crossfit": "crossfit"},
    install_requires=requirements["base"],
    include_package_data=True,
    extras_require={
        **requirements,
        **dev_requirements,
        "all": list(itertools.chain(*list(requirements.values()))),
    },
    python_requires=">=3.7, <3.12",
    test_suite="tests",
)
