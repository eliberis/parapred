from setuptools import setup

setup(
    name="parapred",
    packages=["parapred"],
    entry_points={
        "console_scripts": ['parapred = parapred.parapred:main']
    },
    install_requires=[
        "Keras==2.0.6",
        "pandas>=0.19.2,<0.20",
        "tensorflow==2.5.3",
        "numpy>=1.13",
        "matplotlib>=2.0.0",
        "scikit-learn>=0.18,<0.19",
        "scipy>=0.19",
        "biopython==1.69",
        "docopt>=0.6.2",
        "h5py>=2.6.0",
        "lxml>=4.1.1",
        "requests>=2.18.4"
    ],
    version="1.0.1",
    description="Deep-learning-powered antibody binding site prediction.",
    author="E Liberis",
    author_email="el398@cam.ac.uk",
    url="https://github.com/eliberis/parapred",
    package_data={"parapred": ["data/*.csv", "precomputed/*"]}
)
