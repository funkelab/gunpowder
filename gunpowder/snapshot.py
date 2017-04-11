from batch_filter import BatchFilter
import h5py
import os

class Snapshot(BatchFilter):

    def __init__(self, output_dir='snapshots', skip=100):
        self.output_dir = output_dir
        self.snapshot_num = 0
        self.skip = skip

    def process(self, batch):

        if self.snapshot_num%self.skip == 0:

            try:
                os.mkdir(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, str(self.snapshot_num).zfill(8) + '.hdf')
            with h5py.File(snapshot_name, 'w') as f:
                f['raw'] = batch.raw
                f['raw'].attrs['offset'] = batch.spec.offset
                if batch.gt is not None:
                    f['gt'] = batch.gt
                if batch.gt_mask is not None:
                    f['gt_mask'] = batch.gt_mask

        self.snapshot_num += 1
