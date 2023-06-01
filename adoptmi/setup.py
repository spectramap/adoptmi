import setuptools 


setuptools.setup(
    name="adoptme",
    version="0.0.10",
    author="Juan David Munoz Bolanos",
    author_email="juan.munoz@student.i-med.ac.at",
    description="A package for microscopy/adaptive optical scientists",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "src"},
    py_modules=["optics", "tools", "api"],
    install_requires=["scikit-learn", "scipy", "scikit-image", "matplotlib"],
    packages=setuptools.find_namespace_packages(where='src'),
    python_requires=">=3.8",
)