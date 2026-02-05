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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'astar_planner = planning.nodes.astar_planner_node:main',
            'efe_planner = planning.efe_planner:main',
            'efe1_planner = planning.nodes.efe1_planner_node:main',
            'efe2_planner = planning.nodes.efe2_planner_node:main',
            'efe_ut_planner = planning.nodes.efe_ut_planner_node:main',
            'mpc_planner = planning.nodes.mpc_planner_node:main',
            'pure_efe_planner = planning.pure_efe_planner:main',
            'pure_efe1_planner = planning.nodes.pure_efe1_planner_node:main',
            'pure_efe2_planner = planning.nodes.pure_efe2_planner_node:main',
            'pure_efe_ut_planner = planning.nodes.pure_efe_ut_planner_node:main',
            'pure_mpc_planner = planning.nodes.pure_mpc_planner_node:main',
        ],
    },
)
