from batch_filter import BatchFilter
import h5py
import os

import logging
logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):

    def __init__(self, output_dir='snapshots', every=100):
        self.output_dir = output_dir
        self.every = max(1,every)

    def process(self, batch):

        id = batch.spec.id

        if id%self.every == 0:

            try:
                os.mkdir(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, str(id).zfill(8) + '.hdf')
            logger.debug("saving to " + snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:
                f['volumes/raw'] = batch.raw
                f['volumes/raw'].attrs['offset'] = batch.spec.input_roi.get_offset()
                if batch.gt is not None:
                    f['volumes/labels/neuron_ids'] = batch.gt
                    f['volumes/labels/neuron_ids'].attrs['offset'] = batch.spec.output_roi.get_offset()
                if batch.gt_mask is not None:
                    f['volumes/labels/mask'] = batch.gt_mask
                if batch.gt_affinities is not None:
                    f['volumes/gt_affs'] = batch.gt_affinities
                if batch.prediction is not None:
                    f['volumes/predicted_affs'] = batch.prediction
                if batch.gradient is not None:
                    f['volumes/gradient'] = batch.gradient
                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss
