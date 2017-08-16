import logging
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):
    '''Save a passing batch in an HDF file.

    Args:

        dataset_names (dict): A dictionary from :class:`VolumeType` to names of 
            the datasets to store them in.

        output_dir (string): The directory to save the snapshots. Will be 
            created, if it does not exist.

        output_filename (string): Template for output filenames. '{id}' in the 
            string will be replaced with the ID of the batch. '{iteration}' with 
            the training iteration (if training was performed on this batch).

        every (int): How often to save a batch. 'every=1' indicates that every 
            batch will be stored, 'every=2' every second and so on. By default, 
            every batch will be stored.

        additional_request (:class:`BatchRequest`): An additional batch request 
            to merge with the passing request, if a snapshot is to be made. If 
            not given, only the volumes that are in the batch anyway are 
            recorded.

        compression_type (string or int): Compression strategy.  Legal values 
            are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this 
            indicates gzip compression level. Otherwise, an integer indicates 
            the number of a dynamically loaded compression filter. (See 
            h5py.groups.create_dataset())

        dataset_dtypes (dict): A dictionary from :class:`VolumeType` to datatype
            (eg. np.int8). Volume to store is copied and casted to the specified type.
             Original volume within the pipeline remains unchanged.
        '''

    def __init__(
            self,
            dataset_names,
            output_dir='snapshots',
            output_filename='{id}.hdf',
            every=1,
            additional_request=None,
            compression_type=None,
            dataset_dtypes=None):
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1,every)
        self.additional_request = BatchRequest() if additional_request is None else additional_request
        self.n = 0
        self.compression_type = compression_type
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes

    def prepare(self, request):

        self.record_snapshot = self.n%self.every == 0
        self.n += 1

        # append additional volume requests, don't overwrite existing ones
        for volume_type, roi in self.additional_request.volumes.items():
            if volume_type not in request.volumes:
                request.volumes[volume_type] = roi

    def process(self, batch, request):

        if self.record_snapshot:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, self.output_filename.format(id=str(batch.id).zfill(8),iteration=batch.iteration))
            logging.info('saving to %s' %snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:

                for (volume_type, volume) in batch.volumes.items():

                    if volume_type not in self.dataset_names:
                        continue

                    ds_name = self.dataset_names[volume_type]

                    offset = volume.roi.get_offset()
                    if volume_type in self.dataset_dtypes:
                        dtype = self.dataset_dtypes[volume_type]
                        dataset = f.create_dataset(name=ds_name, data=volume.data.astype(dtype), compression=self.compression_type)
                    else:
                        dataset = f.create_dataset(name=ds_name, data=volume.data, compression=self.compression_type)
                    dataset.attrs['offset'] = offset
                    dataset.attrs['resolution'] = volume_type.voxel_size

                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss

        self.n += 1

