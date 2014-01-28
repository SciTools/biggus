from distutils.core import setup

setup(
    name='Biggus',
    version='0.3',
    url='https://github.com/SciTools/biggus',
    author='Richard Hattersley',
    author_email='rhattersley@gmail.com',
    packages=['biggus', 'biggus.tests'],
    classifiers=['License :: OSI Approved :: '
                 'GNU Lesser General Public License v3 (LGPLv3)'],
    description='Virtual large arrays and lazy evaluation',
    long_description=open('README.rst').read(),
)
