from funlib.persistence.arrays import Array as PersistenceArray
from funlib.geometry import Roi, FloatRoi as FR, FloatCoordinate as FC

from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.batch import Batch

import numpy as np

from .batch_provider import BatchProvider


class ArraySource(BatchProvider):
    """A `array <https://github.com/funkelab/funlib.persistence>`_ source.

    Provides a source for any array that can fit into the funkelab
    funlib.persistence.Array format. This class comes with assumptions about
    the available metadata and convenient methods for indexing the data
    with a :class:`Roi` in world units.

    Args:

        key (:class:`ArrayKey`):

            The ArrayKey for accessing this array.

        array (``Array``):

            A `funlib.persistence.Array` object.

        interpolatable (``bool``, optional):

            Whether the array is interpolatable. If not given it is
            guessed based on dtype.

    """

    def __init__(
        self,
        key: ArrayKey,
        array: PersistenceArray,
        interpolatable: bool | None = None,
    ):
        self.key = key
        self.array = array
        in_roi = self.array.roi
        in_voxel_size = self.array.voxel_size
        rounded_roi = Roi(in_roi.offset.round(), in_roi.shape.round())
        rounded_voxel_size = in_voxel_size.round()
        assert all(np.isclose(in_roi.offset, rounded_roi.offset)), (
            f"ArraySource requires array ROI offset to be an integer. Got {in_roi.offset}."
        )
        assert all(np.isclose(in_roi.shape, rounded_roi.shape)), (
            f"ArraySource requires array ROI shape to be an integer. Got {in_roi.shape}."
        )
        assert all(np.isclose(in_voxel_size, rounded_voxel_size)), (
            f"ArraySource requires array voxel size to be integer. Got {in_voxel_size}."
        )
        self.array_spec = ArraySpec(
            rounded_roi,
            rounded_voxel_size,
            interpolatable,
            False,
            self.array.dtype,
        )

    def setup(self):
        self.provides(self.key, self.array_spec)

    def provide(self, request):
        outputs = Batch()
        out_spec = self.array_spec.copy()
        out_spec.roi = request[self.key].roi
        outputs[self.key] = Array(
            self.array[FR(FC(out_spec.roi.offset), FC(out_spec.roi.shape))], out_spec
        )
        return outputs
