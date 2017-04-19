#!/usr/bin/env python

from distutils.core import setup,Command

setup(name='pycsa',
      version='1.1.1',
      description='Pure Python package for Correlated Substitution Analysis',
      author='Kevin Brown, Christopher Brown',
      author_email='kevin.s.brown@uconn.edu, chris.al.brown@gmail.com',
      url='https://github.com/thelahunginjeet/pycsa',
      packages=['pycsa'],
      package_dir={'pycsa': ''},
      package_data={'pycsa' : ['tests/1iu0.pdb','tests/pdz_test.aln','tests/run_test.py']},
      license='BSD-3',
      classifiers = [
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Programming Language :: Python',
      ],
    )
