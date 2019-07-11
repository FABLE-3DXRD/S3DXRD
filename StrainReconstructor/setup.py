
#!/usr/bin/env python
from __future__ import absolute_import
from setuptools import setup,Extension, find_packages
import sys

setup(
  name='strain_fitter',
  version='0.0.0',
  description='Reconstruction library for scanning 3DXRD',
  license='GNU', maintainer='Axel Henningsson',
  maintainer_email='axel.henningsson@solid.lth.se',
  download_url='',
  url='',
  packages=find_packages(),
  package_dir={"strain_fitter": "strain_fitter"},
)
