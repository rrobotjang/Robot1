from setuptools import setup

package_name = 'vla_orchestrator'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'fastapi', 'uvicorn'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Main orchestration node for VLA system (pick & place workflow)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'orchestrator = vla_orchestrator.orchestrator_node:main',
        ],
    },
)


