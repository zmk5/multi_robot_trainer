"""Setup File for mrt_server."""
from glob import glob
import os
from setuptools import setup


PACKAGE_NAME = 'mrt_server'

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    packages=[PACKAGE_NAME, PACKAGE_NAME + '/policy', PACKAGE_NAME + '/utils'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        (os.path.join('share', PACKAGE_NAME, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', PACKAGE_NAME, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'server_sync_node = mrt_server.sync:main',
            'server_sync_dual_node = mrt_server.sync_dual:main',
            'server_async_node = mrt_server.async:main',
            'server_async_dual_node = mrt_server.async_dual:main',
        ],
    },
)
