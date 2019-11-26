import setuptools

with open("README.md", 'r') as ld:
    long_desc = ld.read()

setuptools.setup(
    name="pavdhutils",
    version="0.0.1",
    author="Paul Vierthaler",
    author_email="vierth@gmail.com",
    description="Collection of utility scripts for DH research",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="none",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha"
    ]
)