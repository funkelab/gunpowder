from batch_filter import BatchFilter
from gunpowder.ext import h5py
import os

import logging
logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):
    '''Save a passing batch in an HDF file.'''

    def __init__(self, output_dir='snapshots', output_filename='{id}.hdf', every=1):
        '''
        output_dir: string

            The directory to save the snapshots. Will be created, if it does not exist.

        output_filename: string

            Template for output filenames. '{id}' in the string will be replaced 
            with the ID of the batch.

        every:

            How often to save a batch. 'every=1' indicates that every batch will 
            be stored, 'every=2' every second and so on. By default, every batch 
            will be stored.
        '''
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1,every)

    def process(self, batch):

        id = batch.spec.id

        if id%self.every == 0:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, self.output_filename.format(id=str(id).zfill(8)))
            logger.info("saving to " + snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:
                input_offset = batch.spec.input_roi.get_offset()
                output_offset = batch.spec.output_roi.get_offset()
                for array_key, array_data, array_offset in [
                    ('volumes/raw', batch.raw, input_offset),
                    ('volumes/labels/neuron_ids', batch.gt, output_offset),
                    ('volumes/labels/mask', batch.gt_mask, output_offset),
                    ('volumes/gt_affs', batch.gt_affinities, output_offset),
                    ('volumes/predicted_affs', batch.prediction, output_offset),
                    ('volumes/gradient', batch.gradient, output_offset),
                ]:
                    if array_data is not None:
                        dataset = f.create_dataset(name=array_key, data=array_data)
                        dataset.attrs['offset'] = array_offset
                        dataset.attrs['resolution'] = batch.spec.resolution
                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss
