import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

from webknossos import Dataset, Mag
from webknossos.dataset.properties import (
    _properties_floating_type_to_python_type
)

logger = logging.getLogger(__name__)


class WKWSource(BatchProvider):
    def __init__(
            self,
            filename,
            datasets,
            array_specs=None,
            mag_specs=None,
            channels_first=True):

        self.filename = filename
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        if mag_specs is None:
            self.mag_specs = {}
        else:
            self.mag_specs = mag_specs

        self.channels_first = channels_first

        # number of spatial dimensions
        self.ndims = None

    def _open_file(self, filename):
        return Dataset(filename)

    def setup(self):
        data_file = self._open_file(self.filename)

        for (array_key, ds_name) in self.datasets.items():

            if ds_name not in data_file.layers:
                raise RuntimeError("%s not in %s" % (ds_name, self.filename))

            spec = self.__read_spec(array_key, data_file, ds_name)

            self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        data_file = self._open_file(self.filename)
        for (array_key, request_spec) in request.array_specs.items():
            logger.debug("Reading %s in %s...", array_key, request_spec.roi)

            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = dataset_roi \
                - self.spec[array_key].roi.get_offset() / voxel_size

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = Array(
                self.__read(
                    data_file,
                    self.datasets[array_key],
                    self.mag_specs[array_key],
                    dataset_roi
                ),
                array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_voxel_size(self, layer, mag_spec):
        mag = Mag(mag_spec).to_np()
        dset_vox_size = np.array(layer.dataset.scale)
        return Coordinate(mag*dset_vox_size)

    def _get_offset(self, layer):
        return Coordinate(layer._properties.bounding_box.topleft)

    def __read_spec(self, array_key, data_file, ds_name):

        layer = data_file.get_layer(ds_name)

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(layer, self.mag_specs[array_key])
            if voxel_size is None:
                voxel_size = Coordinate((1,)*len(layer.dataset.shape))
                logger.warning(
                    "WARNING: File %s does not contain resolution information "
                    "for %s (dataset %s), voxel size has been set to %s. This "
                    "might not be what you want.",
                    self.filename, array_key, ds_name, spec.voxel_size
                )
            spec.voxel_size = voxel_size

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            offset = self._get_offset(layer)
            offset *= spec.voxel_size
            if offset is None:
                offset = Coordinate((0,)*self.ndims)

            shape = Coordinate(layer._properties.bounding_box.size)

            spec.roi = Roi(offset, shape*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == layer._properties.element_class, (
                "dtype %s provided in array_specs for %s, "
                "but differs from dataset %s dtype %s" %
                (
                    self.array_specs[array_key].dtype,
                    array_key, ds_name, layer._properties.element_class
                ))
        else:
            spec.dtype = _properties_floating_type_to_python_type.get(
                layer._properties.element_class,
                layer._properties.element_class
            )

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, ds_name, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, data_file, ds_name, mag, roi):

        array = data_file\
            .get_layer(ds_name)\
            .get_mag(mag)\
            .read(roi.get_offset(), roi.get_shape())

        return array

    def name(self):

        return super().name() + f"[{self.filename}]"
