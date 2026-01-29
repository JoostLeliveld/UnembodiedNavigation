from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sim'
data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),

    # launch files
    (os.path.join('share', package_name, 'launch'),
        glob('launch/*.py')),
]

# install robot_description files
for root, _, files in os.walk('robot_description'):
    data_files.append(
        (os.path.join('share', package_name, root),
         [os.path.join(root, f) for f in files])
    )

# install gazebo_worlds files
for root, _, files in os.walk('gazebo_worlds'):
    data_files.append(
        (os.path.join('share', package_name, root),
         [os.path.join(root, f) for f in files])
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
)
