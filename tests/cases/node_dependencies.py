import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchFilter,
    BatchProvider,
    BatchRequest,
    Roi,
    build,
)


class NodeDependenciesTestSource(BatchProvider):
    def __init__(self, a_key, b_key):
        self.a_key = a_key
        self.b_key = b_key

    def setup(self):
        self.provides(
            self.a_key,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        )

        self.provides(
            self.b_key,
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

    def __init__(self, b_key, c_key):
        self.b_key = b_key
        self.c_key = c_key
        self.context = (20, 20, 20)

    def setup(self):
        self.provides(self.c_key, self.spec[self.b_key])

    def prepare(self, request):
        assert self.c_key in request

        dependencies = BatchRequest()
        dependencies[self.b_key] = ArraySpec(
            request[self.c_key].roi.grow(self.context, self.context)
        )

        return dependencies

    def process(self, batch, request):
        outputs = Batch()

        # make sure a ROI is what we requested
        b_roi = request[self.c_key].roi.grow(self.context, self.context)
        assert batch[self.b_key].spec.roi == b_roi

        # add C to batch
        c = batch[self.b_key].crop(request[self.c_key].roi)
        outputs[self.c_key] = c
        return outputs


def test_dependecies():
    a_key = ArrayKey("A")
    b_key = ArrayKey("B")
    c_key = ArrayKey("C")

    pipeline = NodeDependenciesTestSource(a_key, b_key)
    pipeline += NodeDependenciesTestNode(b_key, c_key)

    c_roi = Roi((100, 100, 100), (100, 100, 100))

    # simple test, ask only for C

    request = BatchRequest()
    request[c_key] = ArraySpec(roi=c_roi)

    with build(pipeline):
        batch = pipeline.request_batch(request)

    assert a_key not in batch
    assert b_key not in batch
    assert batch[c_key].spec.roi == c_roi

    # ask for C and B of same size as needed by node

    b_roi = c_roi.grow((20, 20, 20), (20, 20, 20))

    request = BatchRequest()
    request[c_key] = ArraySpec(roi=c_roi)
    request[b_key] = ArraySpec(roi=b_roi)

    with build(pipeline):
        batch = pipeline.request_batch(request)

    c = batch[c_key]
    b = batch[b_key]
    assert b.spec.roi == b_roi
    assert c.spec.roi == c_roi
    assert np.equal(b.crop(c.spec.roi).data, c.data).all()

    # ask for C and B of larger size

    b_roi = c_roi.grow((40, 40, 40), (40, 40, 40))

    request = BatchRequest()
    request[b_key] = ArraySpec(roi=b_roi)
    request[c_key] = ArraySpec(roi=c_roi)

    with build(pipeline):
        batch = pipeline.request_batch(request)

    b = batch[b_key]
    c = batch[c_key]
    assert a_key not in batch
    assert b.spec.roi == b_roi
    assert c.spec.roi == c_roi
    assert np.equal(b.crop(c.spec.roi).data, c.data).all()

    # ask for C and B of smaller size

    b_roi = c_roi.grow((-40, -40, -40), (-40, -40, -40))

    request = BatchRequest()
    request[b_key] = ArraySpec(roi=b_roi)
    request[c_key] = ArraySpec(roi=c_roi)

    with build(pipeline):
        batch = pipeline.request_batch(request)

    b = batch[b_key]
    c = batch[c_key]
    assert a_key not in batch
    assert b.spec.roi == b_roi
    assert c.spec.roi == c_roi
    assert np.equal(c.crop(b.spec.roi).data, b.data).all()
