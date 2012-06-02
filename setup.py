from os.path import abspath, dirname, join
from setuptools import setup

cwd = abspath(dirname(__file__))
readme = open(join(cwd, 'readme.rst'))
kwds = {'long_description': readme.read()}
readme.close()

setup(name='CSparse.py',
      version='1.0.0',
      description='CSparse.py: a Concise Sparse matrix Python module',
      author='Richard Lincoln',
      author_email='r.w.lincoln@gmail.com',
      url='http://www.cise.ufl.edu/research/sparse/CSparse/',
      install_requires=['numpy'],
      classifiers=['Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Mathematics'],
      py_modules=['csparse'],
      test_suite='csparse_test.main',
      zip_safe=True,
      **kwds)
