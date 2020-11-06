"""Setup File for mrt_worker."""
from glob import glob
import os
from setuptools import setup


PACKAGE_NAME = 'mrt_worker'

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    packages=[PACKAGE_NAME, PACKAGE_NAME + '/policy', PACKAGE_NAME + '/utils'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        (os.path.join('share', PACKAGE_NAME, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', PACKAGE_NAME, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'worker_sync_node = mrt_worker.sync:main',
            'worker_sync_dual_node = mrt.sync_dual:main',
            'worker_async_node = mrt_worker.async:main',
            'tester_node = mrt_worker.test:main',
        ],
    },
)
