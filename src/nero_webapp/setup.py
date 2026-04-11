from glob import glob
import os
from setuptools import find_packages, setup

package_name = "nero_webapp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "static"),
         glob("static/*")),
    ],
    install_requires=["setuptools", "fastapi", "uvicorn"],
    zip_safe=True,
    maintainer="atish",
    maintainer_email="atish@cloudchef.io",
    description="Simple dual-arm Nero webapp (ROS 2 node).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "webapp_node = nero_webapp.webapp_node:main",
        ],
    },
)
