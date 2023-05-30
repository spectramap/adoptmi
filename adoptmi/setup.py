import setuptools 

with open("README.md", "r", enconding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adoptimi",
    version="0.0.3",
    author="Juan David Munoz Bolanos",
    author_email="juan.munoz@student.i-med.ac.at",
    description="A package for microscopy/adaptive optical scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Windows 10",
        "Intended Audience :: Science/Research",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "src"},
    py_modules=["adoptmi"],
    install_requires=["numpy", "matplotlib", "scipy"],
    packages=setuptools.find_namespace_packages(where='src'),
)