from setuptools import find_packages,setup
from typing import List

DOT = "-e . "

def get_requirements(file_path:str)->List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n"," ")for req in requirements]
       
        if DOT in requirements:
            requirements.remove(DOT)
    return requirements

setup(
    name='MLPROJECT',
    version = '0.0.1',
    author = 'Nasim',
    author_email = 'nasimalsajjad@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
    
)