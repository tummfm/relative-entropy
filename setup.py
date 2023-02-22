from setuptools import setup, find_packages

install_requires = [
    'jax>=0.4.3',
    'jax-md>=0.2.5',
    'optax>=0.0.9',
    'dm-haiku>=0.0.9',
    'sympy',
    'cloudpickle',
    'chex',
    'jax-sgmc',
]

extras_requires = {
    'all': ['mdtraj<=1.9.6', 'matplotlib'],
    }

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='rel-entropy',
    version='0.0.1',
    license='Apache 2.0',
    description=('Train neural network potentials via relative entropy'
                 ' and force matching.'),
    author='Stephan Thaler',
    author_email='stephan.thaler@tum.de',
    packages=find_packages(exclude='examples'),
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tummfm/relative-entropy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)
