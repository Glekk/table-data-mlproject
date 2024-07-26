from setuptools import find_packages, setup
from typing import List


E_DOT='-e .'


def get_requirements(file_path:str)->List[str]:
    '''
    Returns the list of requirements
    '''
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if E_DOT in requirements:
            requirements.remove(E_DOT)
    return requirements


setup(
    name='table_data_project',
    version='0.0.1',
    packages=find_packages(),
    requires=get_requirements('requirements.txt')
)