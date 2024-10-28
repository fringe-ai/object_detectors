from setuptools import setup, find_packages

# # Read the contents of your README file
# from pathlib import Path
# this_directory = Path(__file__).parent
# # long_description = (this_directory / "README.md").read_text()

setup(
    name='lmi-ais-utils',  
    version='0.0.1',  
    description='LMI AI Solutions Package',  
    long_description="LMI AI Solutions Package",
    long_description_content_type='text/markdown',
    url='https://github.com/lmitechnologies/LMI_AI_Solutions.git',
    license='MIT',  
    packages=find_packages(),
    keywords='LMI Technologies LMI AI Solutions',  
    python_requires='>=3.8',
)