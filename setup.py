from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='lmdoctor',
    version='0.3.0',    
    description='pkg for extracting and controlling concepts within language models as they generate text',
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