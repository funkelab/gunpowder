from ..batch_provider import BatchProvider
from ..provider_spec import ProviderSpec
from ..batch import Batch
from ..roi import Roi
from ..profiling import Timing
from ..coordinate import Coordinate
import dvision

import logging
logger = logging.getLogger(__name__)


class ReadFailed(Exception):
    pass


class DvidSource(BatchProvider):

    def __init__(self, hostname, port, uuid, raw_array_name, gt_array_name=None):
        """
        :param hostname: str of hostname for DVID server
        :param port: int of port for DVID server
        :param uuid: str of UUID of node on DVID server
        :param raw_array_name: str of data instance for image data
        :param gt_array_name: str of data instance for segmentation label data
        """
        self.hostname = hostname
        self.port = port
        self.url = "http://{}:{}".format(self.hostname, self.port)
        self.uuid = uuid
        self.raw_array_name = raw_array_name
        self.gt_array_name = gt_array_name
        self.node_service = None
        self.dims = 0
        self.spec = ProviderSpec()

    def setup(self):
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
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, array_name)
        info = data_instance.info
        roi_min = info['Extended']['MinPoint']
        if roi_min is not None:
            roi_min = Coordinate(roi_min[::-1])
        roi_max = info['Extended']['MaxPoint']
        if roi_max is not None:
            roi_max = Coordinate(roi_max[::-1])

        return Roi(offset=roi_min, shape=roi_max - roi_min)

    def __read_raw(self, roi):
        slices = roi.get_bounding_box()
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, self.raw_array_name)
        try:
            return data_instance[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading raw at slices {} with {}".format(slices, repr(self))
            raise ReadFailed(msg)

    def __read_gt(self, roi):
        slices = roi.get_bounding_box()
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, self.gt_array_name)
        try:
            return data_instance[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading GT at slices {} with {}".format(slices, repr(self))
            raise ReadFailed(msg)

    def __repr__(self):
        return "DvidSource(hostname={}, port={}, uuid={}, raw_array_name={}, gt_array_name={}".format(
            self.hostname, self.port, self.uuid, self.raw_array_name, self.gt_array_name
        )
