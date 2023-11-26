from setuptools import find_packages, setup
import os
import glob

package_name = 'depthai_test'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            source = os.path.join(path, filename)
            destination_dir = os.path.join('share', package_name, path)
            paths.append((destination_dir, [source]))
    return paths

extra_files = package_files('models')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *extra_files,
        ('share/' + package_name + '/launch', glob.glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depthai_camera_node = depthai_test.depthai_camera_node:main',
            'spatial_object_node = depthai_test.spatial_object_node:main'
        ],
    },
)
