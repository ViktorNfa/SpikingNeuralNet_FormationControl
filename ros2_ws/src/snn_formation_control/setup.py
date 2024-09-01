from setuptools import setup
import os
from glob import glob

package_name = 'snn_formation_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'snn_formation_control.snn_formation_control_node',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ViktorNfa',
    maintainer_email='viktornfa@gmail.com',
    description='Spiking Neural Network Formation Control for TurtleBot3',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'snn_formation_control_node = snn_formation_control.snn_formation_control_node:main'
        ],
    },
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
)