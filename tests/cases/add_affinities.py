from itertools import product

import numpy as np

from gunpowder import (
    AddAffinities,
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Roi,
    build,
)

GT_LABELS = ArrayKey("GT_LABELS")
GT_MASK = ArrayKey("GT_MASK")
GT_AFFINITIES = ArrayKey("GT_AFFINITIES")
GT_AFFINITIES_MASK = ArrayKey("GT_AFFINITIES_MASK")


class ExampleSource(BatchProvider):
    def setup(self):
        self.provides(
            GT_LABELS,
            ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False,
            ),
        )
        self.provides(
            GT_MASK,
            ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False,
            ),
        )

    def provide(self, request):
        batch = Batch()

        roi = request[GT_LABELS].roi
        shape = (roi / self.spec[GT_LABELS].voxel_size).get_shape()
        spec = self.spec[GT_LABELS].copy()
        spec.roi = roi

        batch.arrays[GT_LABELS] = Array(np.random.randint(0, 2, shape), spec)

        roi = request[GT_MASK].roi
        shape = (roi / self.spec[GT_MASK].voxel_size).get_shape()
        spec = self.spec[GT_MASK].copy()
        spec.roi = roi

        batch.arrays[GT_MASK] = Array(np.random.randint(0, 2, shape), spec)

        return batch


def test_output():
    neighborhood = [
        Coordinate((-2, 0, 0)),
        Coordinate((0, -1, 0)),
        Coordinate((0, 0, 1)),
        Coordinate((1, 1, 1)),
    ]

    pipeline = ExampleSource() + AddAffinities(
        neighborhood,
        labels=GT_LABELS,
        labels_mask=GT_MASK,
        affinities=GT_AFFINITIES,
        affinities_mask=GT_AFFINITIES_MASK,
    )

    with build(pipeline):
        for i in range(10):
            request = BatchRequest()
            request.add(GT_LABELS, (100, 16, 64))
            request.add(GT_MASK, (100, 16, 64))
            request.add(GT_AFFINITIES, (100, 16, 64))
            request.add(GT_AFFINITIES_MASK, (100, 16, 64))

            batch = pipeline.request_batch(request)

            assert GT_LABELS in batch.arrays
            assert GT_MASK in batch.arrays
            assert GT_AFFINITIES in batch.arrays
            assert GT_AFFINITIES_MASK in batch.arrays

            labels = batch.arrays[GT_LABELS]
            labels_mask = batch.arrays[GT_MASK]
            affs = batch.arrays[GT_AFFINITIES]
            affs_mask = batch.arrays[GT_AFFINITIES_MASK]

            assert (len(neighborhood),) + labels.data.shape == affs.data.shape

            voxel_roi = Roi((0, 0, 0), labels.data.shape)
            for z, y, x in product(*[range(d) for d in labels.data.shape]):
                p = Coordinate((z, y, x))

                for n in range(len(neighborhood)):
                    pn = p + neighborhood[n]
                    if not voxel_roi.contains(pn):
                        continue

                    a = labels.data[p]
                    b = labels.data[pn]
                    masked = labels_mask.data[p] == 0 or labels_mask.data[pn] == 0

                    if a == b and a != 0 and b != 0:
                        assert affs.data[(n,) + p] == 1.0, (
                            "%s -> %s, %s -> %s, but is not 1" % (p, pn, a, b)
                        )
                    else:
                        assert affs.data[(n,) + p] == 0.0, (
                            "%s -> %s, %s -> %s, but is not 0" % (p, pn, a, b)
                        )
                    if masked:
                        assert affs_mask.data[(n,) + p] == 0.0, (
                            "%s or %s are masked, but mask is not 0" % (p, pn)
                        )

        request = BatchRequest()
        request.add(GT_AFFINITIES, (100, 16, 64))
        request.add(GT_AFFINITIES_MASK, (100, 16, 64))

        batch = pipeline.request_batch(request)

        assert GT_LABELS not in batch.arrays
        assert GT_MASK not in batch.arrays
        assert GT_AFFINITIES in batch.arrays
        assert GT_AFFINITIES_MASK in batch.arrays
