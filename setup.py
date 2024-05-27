#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="WeightedERM",
        version="0.0.1",
        author="Rongguang Wang",
        author_email="rgw@seas.upenn.edu",
        description="Adapting Machine Learning Diagnostic Models to New Populations Using a Small Amount of Data: Results from Clinical Neuroscience",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/rongguangw/weightedERM",
        packages=setuptools.find_packages(),
        classifiers=(
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ),
        install_requires=[
            'numpy>=1.19.4',
            'torch>=1.9.1',
            'scikit-learn>=0.23.2',
            'pandas>=1.1.3',
            'autogluon>=0.8.2',
            'fastai>=2.7.12',
        ]
    )
