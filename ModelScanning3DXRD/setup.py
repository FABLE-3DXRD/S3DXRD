#!/usr/bin/env python
from __future__ import absolute_import
#from distutils.core import setup,Extension
from setuptools import setup
import sys

setup(
  name='ModelScanning3DXRD',
  version='1.0',
  description='Forward model of scanning-3DXRD',
  license='GPL', 
  maintainer='Axel Henningsson',
  maintainer_email='nilsaxelhenningsson@gmail.com',
  url='http://github.com/FABLE-3DXRD/S3DXRD',
  packages=['modelscanning3DXRD'],
  package_dir={"modelscanning3DXRD": "modelscanning3DXRD"},
  scripts=["scripts/ModelScanning3DXRD.py"]
)
