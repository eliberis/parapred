from distutils.core import setup

setup(
    name='parapred',
    version='1.0',
    packages=[''],
    url='https://github.com/eliberis/parapred',
    license='',
    author='E Liberis',
    author_email='el398@cam.ac.uk',
    description='Paratope prediction',
    data_files=[('parapred', ['weights.h5'])],
    scripts=['bin/parapred']
)
