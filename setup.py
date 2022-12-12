from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

with open('requirements.txt', encoding="utf-8") as requirements_file:
    REQUIREMENTS = requirements_file.read().splitlines()

setup(
    name="Harmonization",
    version="0.0.5",
    description="Aligning Human & Machine vision",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas FEL",
    author_email="thomas_fel@brown.edu",
    license="MIT",
    install_requires=REQUIREMENTS,
    extras_require={
        "tests": ["pytest", "pylint"],
        "docs": ["mkdocs", "mkdocs-material", "numkdoc"],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)