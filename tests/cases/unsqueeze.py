import copy

import numpy as np
import pytest

import gunpowder as gp


class ExampleSourceUnsqueeze(gp.BatchProvider):
    def __init__(self, voxel_size, raw_key, labels_key):
        self.raw_key = raw_key
        self.labels_key = labels_key

        self.voxel_size = gp.Coordinate(voxel_size)
        self.roi = gp.Roi((0, 0, 0), (10, 10, 10)) * self.voxel_size

        self.raw = gp.ArrayKey("RAW")
        self.labels = gp.ArrayKey("LABELS")

        self.array_spec_raw = gp.ArraySpec(
            roi=self.roi, voxel_size=self.voxel_size, dtype="uint8", interpolatable=True
        )

        self.array_spec_labels = gp.ArraySpec(
            roi=self.roi,
            voxel_size=self.voxel_size,
            dtype="uint64",
            interpolatable=False,
        )

    def setup(self):
        self.provides(self.raw, self.array_spec_raw)
        self.provides(self.labels, self.array_spec_labels)

    def provide(self, request):
        outputs = gp.Batch()

        # RAW
        raw_spec = copy.deepcopy(self.array_spec_raw)
        raw_spec.roi = request[self.raw].roi

        raw_shape = request[self.raw].roi.shape / self.voxel_size

        outputs[self.raw] = gp.Array(
            np.random.randint(0, 256, raw_shape, dtype=raw_spec.dtype), raw_spec
        )

        # LABELS
        labels_spec = copy.deepcopy(self.array_spec_labels)
        labels_spec.roi = request[self.labels].roi

        labels_shape = request[self.labels].roi.shape / self.voxel_size

        labels = np.ones(labels_shape, dtype=labels_spec.dtype)
        outputs[self.labels] = gp.Array(labels, labels_spec)

        return outputs


def test_unsqueeze():
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate((50, 5, 5))
    input_voxels = gp.Coordinate((10, 10, 10))
    input_size = input_voxels * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = (
        ExampleSourceUnsqueeze(voxel_size, raw, labels)
        + gp.Unsqueeze([raw, labels])
        + gp.Unsqueeze([raw], axis=1)
    )

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
        assert batch[raw].data.shape == (1,) + (1,) + input_voxels
        assert batch[labels].data.shape == (1,) + input_voxels


def test_unsqueeze_not_possible():
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")

    voxel_size = gp.Coordinate((50, 5, 5))
    input_voxels = gp.Coordinate((5, 5, 5))
    input_size = input_voxels * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, input_size)

    pipeline = ExampleSourceUnsqueeze(voxel_size, raw, labels) + gp.Unsqueeze(
        [raw], axis=1
    )

    with pytest.raises(gp.PipelineRequestError):
        with gp.build(pipeline) as p:
            p.request_batch(request)
