import logging

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import dvision
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.profiling import Timing
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)


class DvidSourceReadException(Exception):
    pass

class MaskNotProvidedException(Exception):
    pass


class DvidSource(BatchProvider):

    def __init__(self, hostname, port, uuid, raw_array_name, gt_array_name=None, gt_mask_roi_name=None, resolution=None):
        """
        :param hostname: hostname for DVID server
        :type hostname: str
        :param port: port for DVID server
        :type port: int
        :param uuid: UUID of node on DVID server
        :type uuid: str
        :param raw_array_name: DVID data instance for image data
        :type raw_array_name: str
        :param gt_array_name: DVID data instance for segmentation label data
        :type gt_array_name: str
        :param gt_mask_roi_name: DVID region of interest for masking the segmentation
        :type gt_mask_roi_name: str
        :param resolution: resolution of source voxels in nanometers
        :type resolution: tuple
        """
        self.hostname = hostname
        self.port = port
        self.url = "http://{}:{}".format(self.hostname, self.port)
        self.uuid = uuid
        self.raw_array_name = raw_array_name
        self.gt_array_name = gt_array_name
        self.gt_mask_roi_name = gt_mask_roi_name
        self.specified_resolution = resolution
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
        self.spec.has_gt_mask = self.gt_mask_roi_name is not None

        logger.info("DvidSource.spec:\n{}".format(self.spec))

    def get_spec(self):
        return self.spec

    @property
    def resolution(self):
        if self.specified_resolution is not None:
            return self.specified_resolution
        else:
            fib25_resolution = (8, 8, 8)
            logger.warning("WARNING: your source does not contain resolution information. "
                           "I will assume {}. "
                           "This might not be what you want.".format(fib25_resolution))
            return fib25_resolution

    def request_batch(self, batch_spec):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        if VolumeType.GT_LABELS in batch_spec.with_volumes and not spec.has_gt:
            raise RuntimeError("Asked for GT in a non-GT source.")

        if VolumeType.GT_MASK in batch_spec.with_volumes and not spec.has_gt_mask:
            raise RuntimeError("Asked for GT mask in a source that doesn't have one.")

        input_roi = batch_spec.input_roi
        output_roi = batch_spec.output_roi
        if not self.spec.roi.contains(input_roi):
            raise RuntimeError("Input ROI of batch {} outside of my ROI {}".format(input_roi, self.spec.roi))
        if not self.spec.roi.contains(output_roi):
            raise RuntimeError("Output ROI of batch {} outside of my ROI {}".format(output_roi, self.spec.roi))

        logger.debug("Filling batch request for input {} and output {}".format(input_roi, output_roi))

        batch = Batch(batch_spec)

        # TODO: get resolution from repository
        batch.spec.resolution = self.resolution

        logger.debug("Reading raw...")
        batch.volumes[VolumeType.RAW] = Volume(self.__read_raw(batch_spec.input_roi), interpolate=True)
        if VolumeType.GT_LABELS in batch.spec.with_volumes:
            logger.debug("Reading gt...")
            batch.volumes[VolumeType.GT_LABELS] = Volume(self.__read_gt(batch_spec.output_roi), interpolate=False)
        if VolumeType.GT_MASK in batch.spec.with_volumes:
            batch.volumes[VolumeType.GT_MASK] = Volume(self.__read_gt_mask(batch_spec.output_roi), interpolate=False)
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
            raise DvidSourceReadException(msg)

    def __read_gt(self, roi):
        slices = roi.get_bounding_box()
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, self.gt_array_name)
        try:
            return data_instance[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading GT at slices {} with {}".format(slices, repr(self))
            raise DvidSourceReadException(msg)

    def __read_gt_mask(self, roi):
        """
        :param roi: gunpowder.Roi
        :return: uint8 np.ndarray with roi shape
        """
        if self.gt_mask_roi_name is None:
            raise MaskNotProvidedException
        slices = roi.get_bounding_box()
        dvid_roi = dvision.DVIDRegionOfInterest(self.hostname, self.port, self.uuid, self.gt_mask_roi_name)
        try:
            return dvid_roi[slices]
        except Exception as e:
            print(e)
            msg = "Failure reading GT mask at slices {} with {}".format(slices, repr(self))
            raise DvidSourceReadException(msg)

    def __repr__(self):
        return "DvidSource(hostname={}, port={}, uuid={}, raw_array_name={}, gt_array_name={}".format(
            self.hostname, self.port, self.uuid, self.raw_array_name, self.gt_array_name
        )
