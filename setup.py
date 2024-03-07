from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='lmdoctor',
    version='0.5.1',    
    description='Extract, detect, and control semantic representations within language models as they read and write text',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/joshlevy89/lmdoctor',
    author='Josh Levy',
    author_email='joshlevy89@gmail.com',
    license='MIT',
    packages=['lmdoctor'],
    include_package_data=True,
    install_requires=['jupyter',
                      'ipykernel',
                      'transformers>=4.32.0',
                      'plotly',
                      'tokenizers>=0.13.3',
                      'scikit-learn',
                      'pandas'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.9'
    ],
)