from setuptools import setup

setup(
    name='Biggus',
    version='0.7.0',
    url='https://github.com/SciTools/biggus',
    author='Richard Hattersley',
    author_email='rhattersley@gmail.com',
    packages=['biggus', 'biggus.tests', 'biggus.tests.integration',
              'biggus.tests.unit'],
    classifiers=['License :: OSI Approved :: '
                 'GNU Lesser General Public License v3 (LGPLv3)',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4'],
    description='Virtual large arrays and lazy evaluation',
    long_description=open('README.rst').read(),
    use_2to3=True,
    test_suite='biggus.tests'
)
