from batch_filter import BatchFilter
import h5py
import os

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
            print("Snapshot: saving to " + snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:
                f['volumes/raw'] = batch.raw
                f['volumes/raw'].attrs['offset'] = batch.spec.offset
                if batch.gt is not None:
                    f['volumes/labels/neuron_ids'] = batch.gt
                if batch.gt_mask is not None:
                    f['volumes/labels/mask'] = batch.gt_mask
