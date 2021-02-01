import os
from setuptools import setup
try:
    import base_string as string_types
except ImportError:
    string_types = str

extras_require = {
    'tensorflow': ['tensorflow<2'],
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
        description='Data loading DAG for Greentea.',
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
            "tqdm"
        ],
        extras_require=extras_require,
)
