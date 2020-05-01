# encoding=utf-8
import re
from os.path import join, dirname
from setuptools import setup, find_packages


def read_file_content(filepath):
    with open(join(dirname(__file__), filepath), encoding="utf8") as fp:
        return fp.read()


def find_version(filepath):
    content = read_file_content(filepath)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version(join('gobang', '__init__.py'))
long_description = read_file_content('README.md')

setup(name='gobang',
      version=VERSION,
      url='https://github.com/jidasheng/gobang',
      author='Dasheng Ji',
      author_email='jidasheng@qq.com',
      description='A Gobang(also known as "Five in a Row" and "Gomoku") game equipped with AlphaGo-liked AI.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',

      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          "numpy",
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      entry_points={
          'console_scripts': ['gobang=gobang.app:main'],
      }
      )
