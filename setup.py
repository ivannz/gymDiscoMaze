from setuptools import setup, Extension
from Cython.Build import cythonize

from numpy import get_include

setup(
    name='RandomDiscoMaze',
    description='Randomized Maze for sec.4.1 of https://arxiv.org/abs/2002.06038',
    version='0.6',
    license='MIT',
    packages=[
        'gym_discomaze',  # core implementation
        'gym_discomaze.ext',  # extensions
    ],
    install_requires=[
        'numpy',
        'gym',
        'pyglet',
    ],
    ext_modules=cythonize([
        # perfect maze generator with modern numpy.random API in cython
        Extension(
            'gym_discomaze._maze', [
                'gym_discomaze/_maze.pyx',
            ], extra_compile_args=[
                '-O3', '-Ofast',
            ], include_dirs=[
                get_include(),
            ], define_macros=[
                ('NPY_NO_DEPRECATED_API', 0),
            ],
        ),
    ]),
)
