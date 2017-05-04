from ..batch_provider import BatchProvider
from ..provider_spec import ProviderSpec
from ..batch import Batch
from ..roi import Roi
from ..profiling import Timing
from ..coordinate import Coordinate
from libdvid import DVIDNodeService

import logging
logger = logging.getLogger(__name__)

class ReadFailed(Exception):
    pass

class DvidSource(BatchProvider):

    def __init__(self, url, uuid, raw_array_name, gt_array_name=None):

        self.url = url
        self.uuid = uuid
        self.raw_array_name = raw_array_name
        self.gt_array_name = gt_array_name
        self.node_service = None
        self.dims = 0
        self.spec = ProviderSpec()

    def setup(self):

        logger.info("establishing connection to " + self.url)
        self.node_service = DVIDNodeService(self.url, self.uuid)

        self.spec.roi = self.__get_roi(self.raw_array_name)
        if self.gt_array_name is not None:
            self.spec.gt_roi = self.__get_roi(self.gt_array_name)
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
        batch.raw = self.__read_raw(batch_spec.input_roi)
        if batch.spec.with_gt:
            logger.debug("Reading gt...")
            batch.gt = self.__read_gt(batch_spec.output_roi)
        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __get_roi(self, array_name):

        info = self.node_service.get_typeinfo(array_name)
        roi_min = info['Extended']['MinPoint']
        if roi_min is not None:
            roi_min = Coordinate(roi_min[::-1])
        roi_max = info['Extended']['MaxPoint']
        if roi_max is not None:
            roi_max = Coordinate(roi_max[::-1])

        return Roi(roi_min, roi_max - roi_min)

    def __read_raw(self, roi):

        for i in range(5):
            try:
                return self.node_service.get_gray3D(
                        self.raw_array_name,
                        roi.get_shape(),
                        roi.get_offset(),
                        throttle=False)
            except:
                pass

        raise ReadFailed("Reading raw from DvidSource " + self.url + " failed more than " + str(self.retry) + " times")

    def __read_gt(self, roi):

        for i in range(5):
            try:
                return self.node_service.get_labels3D(
                        self.gt_array_name,
                        roi.get_shape(),
                        roi.get_offset(),
                        throttle=False)
            except:
                pass

        raise ReadFailed("Reading GT from DvidSource " + self.url + " failed more than " + str(self.retry) + " times")
