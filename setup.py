import os
from setuptools import setup
try:
    import base_string as string_types
except ImportError:
    string_types = str

extras_require = {
    'tensorflow': [
        # TF doesn't provide <2.0 wheels for py>=3.8 on pypi
        'tensorflow<2.0; python_version<"3.8"',
        # https://stackoverflow.com/a/72493690
        'protobuf==3.20.*; python_version=="3.7"',
    ],
    'pytorch': ['torch'],
}


dep_set = set()
for value in extras_require.values():
    if isinstance(value, string_types):
        dep_set.add(value)
    else:
        dep_set.update(value)

extras_require['full'] = list(dep_set)

name = 'gunpowder'
here = os.path.abspath(os.path.dirname(__file__))
version_info = {}
with open(os.path.join(here, name, 'version_info.py')) as fp:
    exec(fp.read(), version_info)
version = version_info['_version']


setup(
        name=name,
        version=str(version),
        description='A library to facilitate machine learning on large, multi-dimensional images.',
        url='https://github.com/funkey/gunpowder',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'gunpowder',
            'gunpowder.nodes',
            'gunpowder.tensorflow',
            'gunpowder.tensorflow.nodes',
            'gunpowder.torch',
            'gunpowder.torch.nodes',
            'gunpowder.jax',
            'gunpowder.jax.nodes',
            'gunpowder.contrib',
            'gunpowder.contrib.nodes',
            'gunpowder.ext',
            'gunpowder.zoo',
            'gunpowder.zoo.tensorflow'
        ],
        install_requires=[
            "numpy",
            "scipy",
            "h5py",
            "scikit-image",
            "requests",
            "augment-nd",
            "tqdm",
            "funlib.geometry"
        ],
        extras_require=extras_require,
)
