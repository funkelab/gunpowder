from funlib.persistence.arrays import Array as PersistenceArray

from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.batch import Batch

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
        self.array_spec = ArraySpec(
            self.array.roi,
            self.array.voxel_size,
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
        outputs[self.key] = Array(self.array[out_spec.roi], out_spec)
        return outputs
