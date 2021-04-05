from setuptools import setup

setup(
    name='RandomDiscoMaze',
    description='Randomized Maze for sec.4.1 of https://arxiv.org/abs/2002.06038',
    version='0.3',
    license='MIT',
    packages=[
        'gym_discomaze'
    ],
    install_requires=[
        'numpy',
        'gym',
        'pyglet',
    ]
)
