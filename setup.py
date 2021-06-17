# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from distutils.core import setup
from setuptools import find_packages
import os


extra_files = []
scripts = ['scripts/train_superpixel.py','scripts/train_physio.py','scripts/train_equivariant.py']
package_list = find_packages(exclude=['tests*', 'docker', 'scripts'])

version = os.environ.get('CD_VERSION', '0.1')

setup(name='image_remeshing_cnn',
      version=version,
      packages=package_list,
      scripts=scripts,
      install_requires=['olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
                        'tensorflow-datasets',
                        'medical_ts_datasets @ git+https://github.com/ExpectationMax/medical_ts_datasets.git',
                       'gpytorch','tensorboardX>=1.8','seaborn'],
      package_data={'': extra_files},
      include_package_data=True)
