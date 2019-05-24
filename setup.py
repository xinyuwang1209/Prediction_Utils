'''
setup.py - a setup script
Author: Xinyu Wang
'''

from setuptools import setup, find_packages

import os
import sys
# import BGP_Forecast_Modules

try:
	setup(
		name='Prediction_Utils',
		version='0.0.2',
		author='Xinyu Wang',
		author_email='xinyuwang1209@gmail.com',
		description = ("Prediction_Utils."),
		url='https://github.com/xinyuwang1209/Prediction_Utils.git',
		platforms = 'any',
		classifiers=[
			'Environment :: Console',
			'Intended Audience :: Developers',
			'License :: OSI Approved :: MIT License',
			'Operating System :: OS Independent',
			'Programming Language :: Python',
			'Programming Language :: Python :: 3'
		],
		keywords=['Xinyu, xinyu, pypi, package, rpki'],
		packages=find_packages(include=['Prediction_Utils', 'Prediction_Utils.*']),
		install_requires=[
			'numpy',
			'pandas',
			'pathos',
			'sklearn'
		],

		)
finally:
	pass
