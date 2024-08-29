from funlib.persistence.arrays import Array as PersistenceArray
from gunpowder import Array, ArrayKey, Batch, BatchProvider, ArraySpec


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
    """

    def __init__(
        self,
        key: ArrayKey,
        array: PersistenceArray,
        interpolatable: bool | None = None,
        nonspatial: bool = False,
    ):
        self.key = key
        self.array = array
        self.array_spec = ArraySpec(
            self.array.roi,
            self.array.voxel_size,
            self.interpolatable,
            self.nonspatial,
            self.array.dtype,
        )

        self.interpolatable = interpolatable
        self.nonspatial = nonspatial

    def setup(self):
        self.provides(self.key, self.array_spec)

    def provide(self, request):
        outputs = Batch()
        if self.nonspatial:
            outputs[self.key] = Array(self.array[:], self.array_spec.copy())
        else:
            out_spec = self.array_spec.copy()
            out_spec.roi = request[self.key].roi
            outputs[self.key] = Array(self.array[out_spec.roi], out_spec)
        return outputs
