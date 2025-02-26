# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("secom/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

INSTALL_REQUIRES = [
    "sentence-transformers",
    "omegaconf",
    "vllm",
    "openai",
    "langchain-community",
    "langchain-core",
    "faiss-gpu",
    "rank_bm25",
]

setup(
    name="secom",
    version=VERSION["VERSION"],
    author="The SeCom team",
    author_email="pzs23@mails.tsinghua.edu.cn",
    description="To deliver coherent and personalized experiences in long-term conversations, constructs the memory bank at segment level by introducing a conversation segmentation model, while applying compression based denoising on memory units to enhance memory retrieval.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    keywords="Memory Management, Retrieval-augmented Generation, Prompt Compression, LLMs, Long-term Conversation",
    license="MIT License",
    url="https://github.com/microsoft/SeCom",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
)
