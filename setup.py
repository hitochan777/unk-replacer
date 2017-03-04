from setuptools import setup
import io
import os
import re


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='unk-replacer',
    version=find_version("unk_replacer", "__init__.py"),
    url='https://github.com/hitochan777/unk-replacer',
    author='Hitoshi Otsuki',
    author_email='hitochan777@gmail.com',
    description=('Unknown word replacement tool for Neural Machine Translation'),
    license='MIT',
    packages=['unk_replacer'],
    scripts=['bin/unk-rep'],
    install_requires=['gensim', 'mojimoji', 'dotmap', 'mypy-lang'],
    test_suite='tests'
)
