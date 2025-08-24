import setuptools
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

long_desc = """
Pyamiimage extracts words and line primitives from images using pytesseract and sknw.
"""

setup(
    name='pyamiimage',
    url='https://github.com/petermr/pyamiimage',
    version='0.0.13',
    description='Image analysis for words and graphics.',
    long_description_content_type='text/markdown',
    long_description=readme,
    author="Peter Murray-Rust, Anuv Chakraborty",
    author_email='petermurrayrust@googlemail.com',
    license='Apache2',
    install_requires=required,
    include_package_data=True,
    zip_safe=False,
    keywords='text and image mining',
    package_dir={'pyamiimage': 'pyamiimage'},
    packages=['pyamiimage', 'pyamiimage.wrapper'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pyamiimage=pyamiimage.commandline:main',
        ],
    },
)