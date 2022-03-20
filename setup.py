import setuptools

# python3 setup.py sdist bdist_wheel

setuptools.setup(
    name='cgn',
    version='1.0.0',
    packages=setuptools.find_packages(),
    url='',
    license='MIT',
    author='Fabian Parzer',
    author_email='fabian.kai.parzer@univie.ac.at',
    description='An implementation of the Gauss-Newton method for solving regularized nonlinear least-squares problems'
                'with nonlinear constraints.',
    install_requires=[
        'numpy',
        'scipy',
        'qpsolvers',
        'typing',
        'prettytable'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: GNU GPLv3'
        'Programming Language :: Python :: 3.8',
    ],
)
