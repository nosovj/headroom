"""Build Cython extensions for simhash acceleration."""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="headroom_cython",
    ext_modules=cythonize(
        "headroom/compression/smart/_simhash_cython.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
