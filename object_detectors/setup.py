from setuptools import setup, find_packages

setup(
    name="lmi-ais-object-detectors",
    version="2.0.0",
    description="LMI AIS Object Detectors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="LMI AIS",
    license="MIT",
    python_requires=">=3.7",
    packages=find_packages(where="."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)