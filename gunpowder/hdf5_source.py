import h5py
from batch_provider import BatchProvider
from provider_spec import ProviderSpec
from batch import Batch

class Hdf5Source(BatchProvider):

    def __init__(self, filename, raw_dataset, gt_dataset=None, gt_mask_dataset=None):

        super(Hdf5Source, self).__init__()

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
                self.dims = (min(self.dims[d], dims[d]) for d in range(len(dims)))

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

    def get_spec(self):
        return self.spec

    def request_batch(self, batch_spec):

        if batch_spec.with_gt and not self.has_gt:
            raise RuntimeError("Asked for GT in a non-GT source.")

        if batch_spec.with_gt_mask and not self.has_gt_mask:
            raise RuntimeError("Asked for GT mask in a source that doesn't have one.")

        bb = batch_spec.get_bounding_box()

        print("Filling batch of size %s"%(str(bb)))
        batch = Batch(batch_spec)
        with h5py.File(self.filename, 'r') as f:
            print("Reading raw...")
            batch.raw = self.__read(f, self.raw_dataset, bb)
            if batch.spec.with_gt:
                print("Reading gt...")
                batch.gt = self.__read(f, self.gt_dataset, bb)
            if batch.spec.with_gt_mask:
                print("Reading gt mask...")
                batch.gt_mask = self.__read(f, self.gt_dataset_mask, bb)

        return batch

    def __read(self, f, ds, bb):
        self.__check_bb(f, ds, bb)
        return f[ds][bb]

    def __check_bb(self, f, ds, bb):
        shape = f[ds].shape
        assert(len(shape) == len(bb), "Bounding box %s mismatches dimensions in %s[%s]"%(str(bb), self.filename, ds))
        for d in range(len(shape)):
            assert(bb[d].start >= 0)
            assert(bb[d].stop <= shape[d])
