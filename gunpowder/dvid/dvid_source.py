from ..batch_provider import BatchProvider
from ..provider_spec import ProviderSpec
from ..batch import Batch
from ..roi import Roi
from ..profiling import Timing
from ..coordinate import Coordinate
from diced import DicedStore

import logging
logger = logging.getLogger(__name__)

class DvidSource(BatchProvider):

    stores = {}

    def __init__(self, url, repository, raw_array_name, gt_array_name=None):

        self.url = url
        self.repository_name = repository
        self.raw_array_name = raw_array_name
        self.gt_array_name = gt_array_name
        self.store = None
        self.repository = None
        self.dims = 0
        self.raw_array = None
        self.gt_array = None
        self.spec = ProviderSpec()

    def setup(self):


        if self.url in DvidSource.stores:
            logger.info("re-using existing connection to " + self.url)
            self.store = DvidSource.stores[self.url]
        else:
            logger.info("establishing connection to " + self.url)
            self.store = DicedStore(self.url)
            DvidSource.stores[self.url] = self.store

        self.repository = self.store.open_repo(self.repository_name)

        logger.info("DVID repository contains: " + str(self.repository.list_instances()))
        logger.info("opening " + str(self.raw_array_name) + " for raw")

        self.raw_array = self.repository.get_array(self.raw_array_name)
        self.dims = self.raw_array.get_numdims()

        raw_extents = self.raw_array.get_extents()
        self.spec.roi = Roi(
                (raw_extents[d].start for d in range(self.dims)),
                (raw_extents[d].stop - raw_extents[d].start for d in range(self.dims))
        )

        if self.gt_array_name is not None:

            logger.info("opening " + str(self.gt_array_name) + " for GT")

            self.gt_array = self.repository.get_array(self.gt_array_name)
            assert self.gt_array.get_numdims() == self.dims, "Dimensions of GT and raw differ"

            gt_extents = self.gt_array.get_extents()
            self.spec.gt_roi = Roi(
                    (gt_extents[d].start for d in range(self.dims)),
                    (gt_extents[d].stop - gt_extents[d].start for d in range(self.dims))
            )

            self.spec.has_gt = True

        else:

            self.spec.has_gt = False

        self.spec.has_gt_mask = False

        logger.info("DVID source spec:\n" + str(self.spec))

    def get_spec(self):
        return self.spec

    def request_batch(self, batch_spec):

        timing = Timing(self)
        timing.start()

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

        # TODO: get resolution from repository
        batch.spec.resolution = (1,)*self.dims
        logger.warning("setting resolution to " + str(batch.spec.resolution))

        logger.debug("Reading raw...")
        batch.raw = self.raw_array[batch_spec.input_roi.get_bounding_box()]
        if batch.spec.with_gt:
            logger.debug("Reading gt...")
            batch.gt = self.gt_array[batch_spec.output_roi.get_bounding_box()]
        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
