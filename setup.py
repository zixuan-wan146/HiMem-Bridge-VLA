from setuptools import find_packages, setup


setup(
    name="himem-bridge-vla",
    version="0.1.0",
    description="HiMem-Bridge-VLA: BridgeAttention and hierarchical memory adapters for VLAs",
    packages=find_packages(include=["himem_bridge_vla", "himem_bridge_vla.*", "evaluations", "evaluations.*"]),
    python_requires=">=3.10",
)
