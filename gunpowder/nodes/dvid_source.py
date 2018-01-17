import logging
import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import dvision
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec

logger = logging.getLogger(__name__)

class DvidSource(BatchProvider):
    '''A DVID array source.

    Provides arrays from DVID servers for each array key given.

    Args:

        hostname (string): The name of the DVID server.

        port (int): The port of the DVID server.

        uuid (string): The UUID of the DVID node to use.

        datasets (dict): Dictionary of ArrayKey -> DVID data instance names
            that this source offers.

        masks (dict, optional): Dictionary of ArrayKey -> DVID ROI instance
            names. This will create binary masks from DVID ROIs.

        array_specs (dict, optional): An optional dictionary of
            :class:`ArrayKey` to :class:`ArraySpec` to overwrite the array
            specs automatically determined from the DVID server. This is useful
            to set ``voxel_size``, for example. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            hostname,
            port,
            uuid,
            datasets,
            masks = None,
            array_specs = None):

        self.hostname = hostname
        self.port = port
        self.url = "http://{}:{}".format(self.hostname, self.port)
        self.uuid = uuid

        self.datasets = datasets
        self.masks = masks if masks is not None else {}

        print("Datasets: ", self.datasets)
        print("Masks: ", self.masks)

        self.array_specs = array_specs if array_specs is not None else {}

        self.ndims = None

    def setup(self):

        for array_key, dataset in self.datasets.items():
            spec = self.__get_spec(array_key)
            self.provides(array_key, spec)

        for array_key, dataset in self.masks.items():
            spec = self.__get_mask_spec(array_key)
            self.provides(array_key, spec)

        logger.info("DvidSource.spec:\n{}".format(self.spec))

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for (array_key, request_spec) in request.array_specs.items():

            logger.debug("Reading %s in %s..."%(array_key, request_spec.roi))

            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi/voxel_size

            # shift request roi into dataset
            dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset()/voxel_size

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # read the data
            if array_key in self.datasets:
                data = self.__read_array(self.datasets[array_key], dataset_roi)
            elif array_key in self.masks:
                data = self.__read_mask(self.masks[array_key], dataset_roi)
            else:
                assert False, ("Encountered a request for %s that is neither a volume "
                               "nor a mask."%array_key)

            # add array to batch
            batch.arrays[array_key] = Array(data, array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __get_info(self, array_key):

        if array_key in self.datasets:

            data = dvision.DVIDDataInstance(
                self.hostname,
                self.port,
                self.uuid,
                self.datasets[array_key])

        elif array_key in self.masks:

            data = dvision.DVIDRegionOfInterest(
                self.hostname,
                self.port,
                self.uuid,
                self.masks[array_key])

        else:

            assert False, ("Encountered a request that is neither a volume "
                           "nor a mask.")

        return data.info

    def __get_spec(self, array_key):

        info = self.__get_info(array_key)
        print(info)

        roi_min = info['Extended']['MinPoint']
        if roi_min is not None:
            roi_min = Coordinate(roi_min[::-1])
        roi_max = info['Extended']['MaxPoint']
        if roi_max is not None:
            roi_max = Coordinate(roi_max[::-1])

        data_roi = Roi(
                offset=roi_min,
                shape=(roi_max - roi_min))
        data_dims = Coordinate(data_roi.get_shape())

        if self.ndims is None:
            self.ndims = len(data_dims)
        else:
            assert self.ndims == len(data_dims)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:

            spec.voxel_size = Coordinate((1,)*self.ndims)
            logger.warning("WARNING: DVID instances do not contain resolution "
                           "information. For %s, the voxel size has been set "
                           "to %s. This might not be what you want.",
                           array_key, spec.voxel_size)

        if spec.roi is None:
            spec.roi = data_roi*spec.voxel_size

        data_dtype =  dvision.DVIDDataInstance(
            self.hostname,
            self.port,
            self.uuid,
            self.datasets[array_key]).dtype

        if spec.dtype is not None:
            assert spec.dtype == data_dtype, ("dtype %s provided in array_specs for %s, "
                                              "but differs from instance %s dtype %s"%
                                              (self.array_specs[array_key].dtype,
                                              array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = data_dtype

        if spec.interpolatable is None:

            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8 # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s. "
                           "Based on the dtype %s, it has been set to %s. "
                           "This might not be what you want.",
                           array_key, spec.dtype, spec.interpolatable)

        return spec

    def __get_mask_spec(self, array_key):

        # create initial array spec

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        # get voxel size

        if spec.voxel_size is None:

            spec.voxel_size = Coordinate((1,)*self.ndims)
            logger.warning("WARNING: DVID instances do not contain resolution "
                           "information. For %s, the voxel size has been set "
                           "to %s. This might not be what you want.",
                           array_key, spec.voxel_size)

        # get ROI

        if spec.roi is None:

            for array_key in self.datasets:

                roi = self.spec[array_key].roi

                if spec.roi is None:
                    spec.roi = roi.copy()
                else:
                    spec.roi = roi.union(spec.roi)

        # set interpolatable

        if spec.interpolatable is None:
            spec.interpolatable = False

        # set datatype

        data_dtype = np.uint8

        if spec.dtype is not None:
            assert spec.dtype == data_dtype, ("dtype %s provided in array_specs for %s, "
                                              "but differs from instance %s dtype %s"%
                                              (self.array_specs[array_key].dtype,
                                              array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = data_dtype

        return spec

    def __read_array(self, instance, roi):

        data_instance = dvision.DVIDDataInstance(
            self.hostname,
            self.port,
            self.uuid,
            instance)

        return data_instance[roi.get_bounding_box()]

    def __read_mask(self, instance, roi):

        dvid_roi = dvision.DVIDRegionOfInterest(
            self.hostname,
            self.port,
            self.uuid,
            instance)

        return dvid_roi[roi.get_bounding_box()]

    def __repr__(self):

        return "DvidSource(hostname={}, port={}, uuid={}".format(
            self.hostname,
            self.port,
            self.uuid)
