from __future__ import print_function

import os

from setuptools import setup, Command


class CleanSource(Command):
    """
    Removes orphaned pyc/pyo files from the sources.

    """
    description = 'clean orphaned pyc/pyo files from sources'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root_path, dir_names, file_names in os.walk('biggus'):
            for file_name in file_names:
                if file_name.endswith('pyc') or file_name.endswith('pyo'):
                    compiled_path = os.path.join(root_path, file_name)
                    source_path = compiled_path[:-1]
                    if not os.path.exists(source_path):
                        print('Cleaning', compiled_path)
                        os.remove(compiled_path)


setup(
    name='Biggus',
    version='0.13.0',
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
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5'],
    description='Virtual large arrays and lazy evaluation',
    long_description=open('README.rst').read(),
    use_2to3=True,
    test_suite='biggus.tests',
    cmdclass={'clean_source': CleanSource}
)
