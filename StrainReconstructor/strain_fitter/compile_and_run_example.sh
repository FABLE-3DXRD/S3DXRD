#!/bin/sh
cd ..
rm -r build/ dist/ strain_fitter.egg-info/
python setup.py build install
cd strain_fitter
python example.py
