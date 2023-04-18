from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchFilter,
    BatchRequest,
    Batch,
    ArrayKeys,
    ArraySpec,
    ArrayKey,
    Array,
    Roi,
    build,
)
import numpy as np


class NodeDependenciesTestSource(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.A,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        )

        self.provides(
            ArrayKeys.B,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        )

    def provide(self, request):
        batch = Batch()

        # have the pixels encode their position
        for array_key, spec in request.array_specs.items():
            roi = spec.roi

            for d in range(3):
                assert roi.begin[d] % 4 == 0, "roi %s does not align with voxels"

            data_roi = roi / 4

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(data_roi.begin[0], data_roi.end[0]),
                range(data_roi.begin[1], data_roi.end[1]),
                range(data_roi.begin[2], data_roi.end[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(data, spec)
        return batch


class NodeDependenciesTestNode(BatchFilter):
    """Creates C from B."""

    def __init__(self):
        self.context = (20, 20, 20)

    def setup(self):
        self.provides(ArrayKeys.C, self.spec[ArrayKeys.B])

    def prepare(self, request):
        assert ArrayKeys.C in request

        dependencies = BatchRequest()
        dependencies[ArrayKeys.B] = ArraySpec(
            request[ArrayKeys.C].roi.grow(self.context, self.context)
        )

        return dependencies

    def process(self, batch, request):
        outputs = Batch()

        # make sure a ROI is what we requested
        b_roi = request[ArrayKeys.C].roi.grow(self.context, self.context)
        assert batch[ArrayKeys.B].spec.roi == b_roi

        # add C to batch
        c = batch[ArrayKeys.B].crop(request[ArrayKeys.C].roi)
        outputs[ArrayKeys.C] = c
        return outputs


class TestNodeDependencies(ProviderTest):
    def test_dependecies(self):
        ArrayKey("A")
        ArrayKey("B")
        ArrayKey("C")

        pipeline = NodeDependenciesTestSource()
        pipeline += NodeDependenciesTestNode()

        c_roi = Roi((100, 100, 100), (100, 100, 100))

        # simple test, ask only for C

        request = BatchRequest()
        request[ArrayKeys.C] = ArraySpec(roi=c_roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        assert ArrayKeys.A not in batch
        assert ArrayKeys.B not in batch
        assert batch[ArrayKeys.C].spec.roi == c_roi

        # ask for C and B of same size as needed by node

        b_roi = c_roi.grow((20, 20, 20), (20, 20, 20))

        request = BatchRequest()
        request[ArrayKeys.C] = ArraySpec(roi=c_roi)
        request[ArrayKeys.B] = ArraySpec(roi=b_roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        c = batch[ArrayKeys.C]
        b = batch[ArrayKeys.B]
        assert b.spec.roi == b_roi
        assert c.spec.roi == c_roi
        assert np.equal(b.crop(c.spec.roi).data, c.data).all()

        # ask for C and B of larger size

        b_roi = c_roi.grow((40, 40, 40), (40, 40, 40))

        request = BatchRequest()
        request[ArrayKeys.B] = ArraySpec(roi=b_roi)
        request[ArrayKeys.C] = ArraySpec(roi=c_roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        b = batch[ArrayKeys.B]
        c = batch[ArrayKeys.C]
        assert ArrayKeys.A not in batch
        assert b.spec.roi == b_roi
        assert c.spec.roi == c_roi
        assert np.equal(b.crop(c.spec.roi).data, c.data).all()

        # ask for C and B of smaller size

        b_roi = c_roi.grow((-40, -40, -40), (-40, -40, -40))

        request = BatchRequest()
        request[ArrayKeys.B] = ArraySpec(roi=b_roi)
        request[ArrayKeys.C] = ArraySpec(roi=c_roi)

        with build(pipeline):
            batch = pipeline.request_batch(request)

        b = batch[ArrayKeys.B]
        c = batch[ArrayKeys.C]
        assert ArrayKeys.A not in batch
        assert b.spec.roi == b_roi
        assert c.spec.roi == c_roi
        assert np.equal(c.crop(b.spec.roi).data, b.data).all()
