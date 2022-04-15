# -*- coding: utf-8 -*-
#
# @Time    : 2022-04-15 14:33:31
# @Author  : Yunlong Shi
# @Email   : syljoy@163.com
# @FileName: setup.py
# @Software: PyCharm
# @Github  : https://github.com/syljoy
# @Desc    :


from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='PyTorch-CQRN',
    version='0.1.1',
    description="Convolutional Quasi-Recurrent Network (CQRN)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/syljoy/pytorch-cqrn",
    author="syljoy",
    author_email='syljoy@163.com',
    packages=['torchcqrn', ],
    license='MIT License',
    python_requires='>=3',
)
