from setuptools import setup

setup(
    name='j2learn',
    version='0.1.0',
    description='A package',
    url='https://github.com/jwoz/j2learn',
    author='JW',
    author_email='',
    packages=['j2learn'],
    install_requires=[
        'pandas',
        'numpy',
        'python-mnist',
    ],
    classifiers=[
        'Development Status :: Prototyping',
        'Programming Language :: Python :: 3.8',
    ],
)
