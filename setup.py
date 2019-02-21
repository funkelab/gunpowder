from setuptools import setup
try:
    import base_string as string_types
except ImportError:
    string_types = str

extras_require = {
    'tensorflow': ['tensorflow'],
}


dep_set = set()
for value in extras_require.values():
    if isinstance(value, string_types):
        dep_set.add(value)
    else:
        dep_set.update(value)

extras_require['full'] = list(dep_set)

setup(
        name='gunpowder',
        version='0.3.2',
        description='Data loading DAG for Greentea.',
        url='https://github.com/funkey/gunpowder',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'gunpowder',
            'gunpowder.nodes',
            'gunpowder.caffe',
            'gunpowder.caffe.nodes',
            'gunpowder.tensorflow',
            'gunpowder.tensorflow.nodes',
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
            "augment-nd"
        ],
        extras_require=extras_require,
)
