from setuptools import setup
import os
from glob import glob

package_name = 'visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        # Package index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # Package manifest
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        # RViz config files
        (os.path.join('share', package_name, 'rviz'),
         glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='Centralized RViz configuration for thesis workspace',
    license='MIT',
)
