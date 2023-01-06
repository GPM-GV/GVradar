from distutils.core import setup

setup(
    name='GVradar',
    version='1.0',
    author='Jason Pippitt',
    author_email='jason.l.pippitt@nasa.gov',
    packages=['', 'gvradar'],
    scripts=[],
    license='LICENSE.txt',
    description='Dual Pol Quality Control and precipitation product package which utilizes the Python ARM Radar Toolkit and CSU Radar Tools',
    long_description=open('README.rst').read(),
)
