import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="energize",
    version="0.1",
    author="Gabriel Cortês",
    author_email="cortes@dei.uc.pt",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RodriguesGabriel/energize",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.9',
    # install_requires=requirements,
)
