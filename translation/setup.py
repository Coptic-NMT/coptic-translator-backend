from setuptools import setup, find_packages

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'translation'))

setup(
    name='translation',
    version='0.1',
    packages=find_packages(),
)



