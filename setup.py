from setuptools import setup
import os

VERSION = "0.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


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
    install_requires=[],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.7",
)
