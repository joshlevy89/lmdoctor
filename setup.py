from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='lmdoctor',
    version='0.5.7',    
    description='Extract, detect, and control representations within language models as they read and write text.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/joshlevy89/lmdoctor',
    author='Josh Levy',
    author_email='joshlevy89@gmail.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    # versions known to work with auto-gptq
    install_requires=['jupyter',
                      'ipykernel',
                      'transformers==4.36.2',
                      'plotly',
                      'tokenizers==0.15.2',
                      'peft==0.10.0',
                      'scikit-learn',
                      'pandas'
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10'
    ],
)