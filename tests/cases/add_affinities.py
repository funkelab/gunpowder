from itertools import product

import numpy as np

from gunpowder import (
    AddAffinities,
    Array,
    ArrayKey,
    ArrayKeys,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Roi,
    build,
)


class ExampleSource(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False,
            ),
        )
        self.provides(
            ArrayKeys.GT_MASK,
            ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False,
            ),
        )

    def provide(self, request):
        batch = Batch()

        roi = request[ArrayKeys.GT_LABELS].roi
        shape = (roi / self.spec[ArrayKeys.GT_LABELS].voxel_size).get_shape()
        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi

        batch.arrays[ArrayKeys.GT_LABELS] = Array(np.random.randint(0, 2, shape), spec)

        roi = request[ArrayKeys.GT_MASK].roi
        shape = (roi / self.spec[ArrayKeys.GT_MASK].voxel_size).get_shape()
        spec = self.spec[ArrayKeys.GT_MASK].copy()
        spec.roi = roi

        batch.arrays[ArrayKeys.GT_MASK] = Array(np.random.randint(0, 2, shape), spec)

        return batch


def test_output():
    labels_key = ArrayKey("GT_LABELS")
    mask_key = ArrayKey("GT_MASK")
    affs_key = ArrayKey("GT_AFFINITIES")
    affs_mask_key = ArrayKey("GT_AFFINITIES_MASK")

    neighborhood = [
        Coordinate((-2, 0, 0)),
        Coordinate((0, -1, 0)),
        Coordinate((0, 0, 1)),
        Coordinate((1, 1, 1)),
    ]

    pipeline = ExampleSource() + AddAffinities(
        neighborhood,
        labels=labels_key,
        labels_mask=mask_key,
        affinities=affs_key,
        affinities_mask=affs_mask_key,
    )

    with build(pipeline):
        for i in range(10):
            request = BatchRequest()
            request.add(labels_key, (100, 16, 64))
            request.add(mask_key, (100, 16, 64))
            request.add(affs_key, (100, 16, 64))
            request.add(affs_mask_key, (100, 16, 64))

            batch = pipeline.request_batch(request)

            assert labels_key in batch.arrays
            assert mask_key in batch.arrays
            assert affs_key in batch.arrays
            assert affs_mask_key in batch.arrays

            labels = batch.arrays[labels_key]
            labels_mask = batch.arrays[mask_key]
            affs = batch.arrays[affs_key]
            affs_mask = batch.arrays[affs_mask_key]

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
                        assert (
                            affs.data[(n,) + p] == 1.0
                        ), "%s -> %s, %s -> %s, but is not 1" % (p, pn, a, b)
                    else:
                        assert (
                            affs.data[(n,) + p] == 0.0
                        ), "%s -> %s, %s -> %s, but is not 0" % (p, pn, a, b)
                    if masked:
                        assert (
                            affs_mask.data[(n,) + p] == 0.0
                        ), "%s or %s are masked, but mask is not 0" % (p, pn)

        request = BatchRequest()
        request.add(affs_key, (100, 16, 64))
        request.add(affs_mask_key, (100, 16, 64))

        batch = pipeline.request_batch(request)

        assert labels_key not in batch.arrays
        assert mask_key not in batch.arrays
        assert affs_key in batch.arrays
        assert affs_mask_key in batch.arrays
