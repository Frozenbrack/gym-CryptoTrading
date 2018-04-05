from setuptools.extension import Extension
from Cython.Build import cythonize, build_ext
from setuptools import setup, find_packages

import numpy

cmdclass = {}

extensions = [
  Extension(
     "gym_CryptoTrading.envs.modules.sampler",
     [
         "gym_CryptoTrading/envs/modules/sampler.pyx"
     ],
     include_dirs=[numpy.get_include()]
  )
]

cmdclass.update({ 'build_ext': build_ext })

setup(name='gym_CryptoTrading',
      version='0.0.1',
      install_requires=['gym', 'cython'],
      packages = find_packages(),
      ext_modules=cythonize(extensions),
)  
