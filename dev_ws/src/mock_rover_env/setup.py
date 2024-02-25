from setuptools import find_packages, setup

package_name = 'mock_rover_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devlos',
    maintainer_email='18aguilerac@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_feed = mock_rover_env.camera_feed:main'
            'object_detection = mock_rover_env.object_detection:main'
        ],
    },
)
