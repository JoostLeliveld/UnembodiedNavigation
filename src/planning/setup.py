from setuptools import find_packages, setup

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'scipy'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='Visibility-aware active planning and runtime nodes for UnembodiedNavigation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'efe_planner = planning.nodes.efe_planner_node:main',
            'mpc_planner = planning.nodes.mpc_planner_node:main',
            'efer_planner = planning.nodes.efer_planner_node:main',
            'efe_agent = planning.nodes.efe_agent_node:main',
        ],
    },
)
