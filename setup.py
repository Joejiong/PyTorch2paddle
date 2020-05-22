from setuptools import setup

setup(
    name='Joe_nn_transfer',
    version='0.1.0',
    description='Transfer weights between PaddlePaddle and Keras and PyTorch.',
    install_requires=[
        'numpy',
        'keras',
        'h5py',
        'paddlepaddle'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['Joe_nn_transfer'],
)
