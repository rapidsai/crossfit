import codecs
import itertools
import os

from setuptools import setup

VERSION = "0.0.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


def read_requirements(filename):
    base = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(base, filename), "rb", "utf-8") as f:
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith("#")]


_dev = read_requirements("requirements/dev.txt")

requirements = {
    "base": read_requirements("requirements/base.txt"),
    "dev": _dev,
    "tensorflow": read_requirements("requirements/tensorflow.txt"),
    "pytorch": read_requirements("requirements/pytorch.txt"),
    "jax": read_requirements("requirements/jax.txt"),
}
dev_requirements = {
    "tensorflow-dev": requirements["tensorflow"] + _dev,
    "pytorch-dev": requirements["pytorch"] + _dev,
    "jax-dev": requirements["jax"] + _dev,
}


setup(
    name="crossfit",
    description="Metric calculation library",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Marc Romeyn",
    url="https://github.com/marcromeyn/crossfit",
    project_urls={
        "Issues": "https://github.com/marcromeyn/crossfit/issues",
        "CI": "https://github.com/marcromeyn/crossfit/actions",
        "Changelog": "https://github.com/marcromeyn/crossfit/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["crossfit"],
    install_requires=requirements["base"],
    include_package_data=True,
    extras_require={
        **requirements,
        **dev_requirements,
        "all": list(itertools.chain(*list(requirements.values()))),
    },
    python_requires=">=3.7",
    test_suite="tests",
)
