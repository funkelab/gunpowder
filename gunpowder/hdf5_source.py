import h5py
import numpy as np
from batch_provider import BatchProvider
from provider_spec import ProviderSpec
from batch import Batch
from roi import Roi

import logging
logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):

    def __init__(self, filename, raw_dataset, gt_dataset=None, gt_mask_dataset=None):

        self.filename = filename
        self.raw_dataset = raw_dataset
        self.gt_dataset = gt_dataset
        self.gt_mask_dataset = gt_mask_dataset

        f = h5py.File(filename, 'r')

        self.dims = None
        for ds in [raw_dataset, gt_mask_dataset, gt_mask_dataset]:

            if ds is None:
                continue

            if ds not in f:
                raise RuntimeError("%s not in %s"%(ds,filename))

            if self.dims is None:
                self.dims = f[ds].shape
            else:
                dims = f[ds].shape
                assert(len(dims) == len(self.dims))
                self.dims = tuple(min(self.dims[d], dims[d]) for d in range(len(dims)))

        if 'resolution' not in f[raw_dataset].attrs:
            logger.warning("WARNING: your source does not contain resolution information (no attribute 'resolution' in raw dataset). I will assume (1,1,1). This might not be what you want.")

        f.close()

        self.spec = ProviderSpec()
        self.spec.roi = Roi(
                (0,)*len(self.dims),
                self.dims
        )
        self.spec.has_gt = gt_dataset is not None
        self.spec.has_gt_mask = gt_mask_dataset is not None

    def get_spec(self):
        return self.spec

    def request_batch(self, batch_spec):

        spec = self.get_spec()

        if batch_spec.with_gt and not spec.has_gt:
            raise RuntimeError("Asked for GT in a non-GT source.")

        if batch_spec.with_gt_mask and not spec.has_gt_mask:
            raise RuntimeError("Asked for GT mask in a source that doesn't have one.")

        input_roi = batch_spec.input_roi
        output_roi = batch_spec.output_roi
        if not self.spec.roi.contains(input_roi):
            raise RuntimeError("Input ROI of batch %s outside of my ROI %s"%(input_roi,self.spec.roi))
        if not self.spec.roi.contains(output_roi):
            raise RuntimeError("Output ROI of batch %s outside of my ROI %s"%(output_roi,self.spec.roi))

        logger.debug("Filling batch request for input %s and output %s"%(str(input_roi),str(output_roi)))

        batch = Batch(batch_spec)
        with h5py.File(self.filename, 'r') as f:
            if 'resolution' in f[self.raw_dataset].attrs:
                batch.spec.resolution = tuple(f[self.raw_dataset].attrs['resolution'])
                logger.debug("providing batch with resolution of " + str(batch.spec.resolution))
            else:
                batch.spec.resolution = (1,1,1)
            logger.debug("Reading raw...")
            batch.raw = self.__read(f, self.raw_dataset, input_roi)
            if batch.spec.with_gt:
                logger.debug("Reading gt...")
                batch.gt = self.__read(f, self.gt_dataset, output_roi)
            if batch.spec.with_gt_mask:
                logger.debug("Reading gt mask...")
                batch.gt_mask = self.__read(f, self.gt_mask_dataset, output_roi)

        logger.debug("done")
        return batch

    def __read(self, f, ds, roi):

        return np.array(f[ds][roi.get_bounding_box()])
