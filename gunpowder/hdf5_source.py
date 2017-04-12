import h5py
import numpy as np
from batch_provider import BatchProvider
from provider_spec import ProviderSpec
from batch import Batch

class Hdf5Source(BatchProvider):

    def __init__(self, filename, raw_dataset, gt_dataset=None, gt_mask_dataset=None):

        f = h5py.File(filename, 'r')

        self.dims = None
        self.spec = ProviderSpec()
        for ds in [raw_dataset, gt_mask_dataset, gt_mask_dataset]:

            if ds is None:
                continue

            if ds not in f:
                raise RuntimeError("%s not in %s"%(raw_dataset,filename))

            if self.dims is None:
                self.dims = f[ds].shape
            else:
                dims = f[ds].shape
                assert(len(dims) == len(self.dims))
                self.dims = tuple(min(self.dims[d], dims[d]) for d in range(len(dims)))

        f.close()

        self.spec.bounding_box = tuple(
                slice(0, self.dims[d])
                for d in range(len(self.dims))
        )
        self.spec.has_gt = gt_dataset is not None
        self.spec.has_gt_mask = gt_mask_dataset is not None

        self.filename = filename
        self.raw_dataset = raw_dataset
        self.gt_dataset = gt_dataset
        self.gt_mask_dataset = gt_mask_dataset

    def get_spec(self):
        return self.spec

    def request_batch(self, batch_spec):

        spec = self.get_spec()

        if batch_spec.with_gt and not spec.has_gt:
            raise RuntimeError("Asked for GT in a non-GT source.")

        if batch_spec.with_gt_mask and not spec.has_gt_mask:
            raise RuntimeError("Asked for GT mask in a source that doesn't have one.")

        bb = batch_spec.get_bounding_box()
        common_bb = self.__intersect(bb, spec.get_bounding_box())

        print("Hdf5Source: Filling batch request for %s with data from %s"%(str(bb),str(common_bb)))
        batch = Batch(batch_spec)
        with h5py.File(self.filename, 'r') as f:
            if 'resolution' in f[self.raw_dataset].attrs:
                batch.spec.resolution = tuple(f[self.raw_dataset].attrs['resolution'])
                print("Hdf5Source: providing batch with resolution of " + str(batch.spec.resolution))
            else:
                print("Hdf5Source: WARNING: your source does not contain resolution information (no attribute 'resolution' in raw dataset). I will assume (1,1,1). This might not be what you want.")
                batch.spec.resolution = (1,1,1)
            print("Hdf5Source: Reading raw...")
            batch.raw = self.__read(f, self.raw_dataset, bb, common_bb)
            if batch.spec.with_gt:
                print("Hdf5Source: Reading gt...")
                batch.gt = self.__read(f, self.gt_dataset, bb, common_bb)
            if batch.spec.with_gt_mask:
                print("Hdf5Source: Reading gt mask...")
                batch.gt_mask = self.__read(f, self.gt_mask_dataset, bb, common_bb)

        return batch

    def __intersect(self, bb1, bb2):
        return tuple(
                slice(max(bb1[d].start, bb2[d].start),min(bb1[d].stop, bb2[d].stop))
                for d in range(len(bb1))
        )

    def __read(self, f, ds, target_bb, data_bb, no_data_value=0):
        self.__check_bb(f, ds, data_bb)
        a = np.zeros(
                tuple(target_bb[d].stop - target_bb[d].start for d in range(len(target_bb))),
                dtype=f[ds].dtype
        )
        if no_data_value != 0:
            a[:] = no_data_value
        data_in_target = tuple(
                slice(data_bb[d].start - target_bb[d].start, data_bb[d].stop - target_bb[d].start)
                for d in range(len(target_bb))
        )
        a[data_in_target] = f[ds][data_bb]
        return a

    def __check_bb(self, f, ds, bb):
        shape = f[ds].shape
        assert len(shape) == len(bb), "Bounding box %s mismatches dimensions in %s[%s]"%(str(bb), self.filename, ds)
        for d in range(len(shape)):
            assert bb[d].start >= 0, "Bounding box not contained in volume"
            assert bb[d].stop <= shape[d], "Bounding box not contained in volume"
