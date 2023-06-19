import setuptools 

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="adoptmi",
    version="0.0.14",
    author="Juan David Munoz Bolanos",
    author_email="juan.munoz@student.i-med.ac.at",
    description="A package for microscopy/adaptive optical scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=["scikit-learn", "scipy", "scikit-image", "matplotlib"],
    packages=setuptools.find_namespace_packages(where='src'),
    python_requires=">=3.8",
)