import logging
import numpy as np
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py

logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):
    '''Save a passing batch in an HDF file.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the snapshots. Will be created, if it does
            not exist.

        output_filename (``string``):

            Template for output filenames. ``{id}`` in the string will be
            replaced with the ID of the batch. ``{iteration}`` with the training
            iteration (if training was performed on this batch).

        every (``int``):

            How often to save a batch. ``every=1`` indicates that every batch
            will be stored, ``every=2`` every second and so on. By default,
            every batch will be stored.

        additional_request (:class:`BatchRequest`):

            An additional batch request to merge with the passing request, if a
            snapshot is to be made. If not given, only the arrays that are in
            the batch anyway are recorded. This is useful to request additional
            arrays like loss gradients for visualization that are otherwise not
            needed.

        compression_type (``string`` or ``int``):

            Compression strategy.  Legal values are ``gzip``, ``szip``,
            ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
            compression level.

        dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

            A dictionary from array keys to datatype (eg. ``np.int8``). If
            given, arrays are stored using this type. The original arrays
            within the pipeline remain unchanged.

        store_value_range (``bool``):

            If set to ``True`, store range of values in data set attributes.
        '''

    def __init__(
            self,
            dataset_names,
            output_dir='snapshots',
            output_filename='{id}.hdf',
            every=1,
            additional_request=None,
            compression_type=None,
            dataset_dtypes=None,
            store_value_range=False):
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1,every)
        self.additional_request = BatchRequest() if additional_request is None else additional_request
        self.n = 0
        self.compression_type = compression_type
        self.store_value_range = store_value_range
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes

    def prepare(self, request):

        self.record_snapshot = self.n%self.every == 0

        # append additional array requests, don't overwrite existing ones
        for array_key, spec in self.additional_request.array_specs.items():
            if array_key not in request.array_specs:
                request.array_specs[array_key] = spec

    def process(self, batch, request):

        if self.record_snapshot:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(
                self.output_dir,
                self.output_filename.format(
                    id=str(batch.id).zfill(8),
                    iteration=int(batch.iteration or 0)))
            logger.info('saving to %s' %snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:

                for (array_key, array) in batch.arrays.items():

                    if array_key not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[array_key]

                    if array_key in self.dataset_dtypes:
                        dtype = self.dataset_dtypes[array_key]
                        dataset = f.create_dataset(name=ds_name, data=array.data.astype(dtype), compression=self.compression_type)
                    else:
                        dataset = f.create_dataset(name=ds_name, data=array.data, compression=self.compression_type)
                    
                    if array.spec.roi is not None:
                        dataset.attrs['offset'] = array.spec.roi.get_offset()
                    dataset.attrs['resolution'] = self.spec[array_key].voxel_size

                    if self.store_value_range:
                        dataset.attrs['value_range'] = (
                            np.asscalar(array.data.min()),
                            np.asscalar(array.data.max()))

                    # if array has attributes, add them to the dataset
                    for attribute_name, attribute in array.attrs.items():
                        dataset.attrs[attribute_name] = attribute

                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss

        self.n += 1

