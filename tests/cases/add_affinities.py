from .provider_test import ProviderTest
from gunpowder import *
from itertools import product
from unittest import skipIf
import itertools
import numpy as np
import logging


class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.GT_LABELS, ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))
        self.provides(
            ArrayKeys.GT_MASK, ArraySpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))

    def provide(self, request):

        batch = Batch()

        roi = request[ArrayKeys.GT_LABELS].roi
        shape = (roi/self.spec[ArrayKeys.GT_LABELS].voxel_size).get_shape()
        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi

        batch.arrays[ArrayKeys.GT_LABELS] = Array(
            np.random.randint(
                0, 2,
                shape
            ),
            spec
        )

        roi = request[ArrayKeys.GT_MASK].roi
        shape = (roi/self.spec[ArrayKeys.GT_MASK].voxel_size).get_shape()
        spec = self.spec[ArrayKeys.GT_MASK].copy()
        spec.roi = roi

        batch.arrays[ArrayKeys.GT_MASK] = Array(
            np.random.randint(
                0, 2,
                shape
            ),
            spec
        )

        return batch


class TestAddAffinities(ProviderTest):

    @skipIf(isinstance(gunpowder.ext.malis, gunpowder.ext.NoSuchModule), "malis not installed")
    def test_output(self):

        neighborhood = [
                Coordinate((-2,0,0)),
                Coordinate((0,-1,0)),
                Coordinate((0,0,1)),
                Coordinate((1,1,1))
        ]

        pipeline = (
                TestSource() +
                AddAffinities(
                    neighborhood,
                    labels=ArrayKeys.GT_LABELS,
                    labels_mask=ArrayKeys.GT_MASK,
                    affinities=ArrayKeys.GT_AFFINITIES,
                    affinities_mask=ArrayKeys.GT_AFFINITIES_MASK)
        )

        with build(pipeline):

            for i in range(10):

                request = BatchRequest()
                request.add(ArrayKeys.GT_LABELS, (100,16,64))
                request.add(ArrayKeys.GT_MASK, (100,16,64))
                request.add(ArrayKeys.GT_AFFINITIES, (100,16,64))
                request.add(ArrayKeys.GT_AFFINITIES_MASK, (100,16,64))

                batch = pipeline.request_batch(request)

                self.assertTrue(ArrayKeys.GT_LABELS in batch.arrays)
                self.assertTrue(ArrayKeys.GT_MASK in batch.arrays)
                self.assertTrue(ArrayKeys.GT_AFFINITIES in batch.arrays)
                self.assertTrue(ArrayKeys.GT_AFFINITIES_MASK in batch.arrays)

                labels = batch.arrays[ArrayKeys.GT_LABELS]
                labels_mask = batch.arrays[ArrayKeys.GT_MASK]
                affs = batch.arrays[ArrayKeys.GT_AFFINITIES]
                affs_mask = batch.arrays[ArrayKeys.GT_AFFINITIES_MASK]

                self.assertTrue((len(neighborhood),) + labels.data.shape == affs.data.shape)

                voxel_roi = Roi((0,0,0), labels.data.shape)
                for (z,y,x) in product(*[range(d) for d in labels.data.shape]):

                    p = Coordinate((z,y,x))

                    for n in range(len(neighborhood)):

                        pn = p + neighborhood[n]
                        if not voxel_roi.contains(pn):
                            continue

                        a = labels.data[p]
                        b = labels.data[pn]
                        masked = (
                            labels_mask.data[p] == 0 or
                            labels_mask.data[pn] == 0)

                        if a == b and a != 0 and b != 0:
                            self.assertEqual(affs.data[(n,)+p], 1.0, "%s -> %s, %s -> %s, but is not 1"%(p, pn, a, b))
                        else:
                            self.assertEqual(affs.data[(n,)+p], 0.0, "%s -> %s, %s -> %s, but is not 0"%(p, pn, a, b))
                        if masked:
                            self.assertEqual(affs_mask.data[(n,)+p], 0.0, (
                                "%s or %s are masked, but mask is not 0"%
                                (p, pn)))
