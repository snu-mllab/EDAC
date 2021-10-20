from distutils.core import setup
from setuptools import find_packages

setup(
    name='lifelong_rl',
    version='0.1',
    intall_requires=['numpy',
                     'gym',
                     'mujoco_py',
                     'd4rl'],
    packages=find_packages(),
)

