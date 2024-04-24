import numpy as np

from gunpowder import Array, ArrayKey, ArraySpec, BatchRequest, Roi, build
from gunpowder.contrib import AddBoundaryDistanceGradients

from .helper_sources import ArraySource


def test_output():
    labels_key = ArrayKey("LABELS")
    dist_key = ArrayKey("BOUNDARY_DISTANCES")
    grad_key = ArrayKey("BOUNDARY_GRADIENTS")

    labels_spec = ArraySpec(
        roi=Roi((0, 0, 0), (120, 16, 64)),
        voxel_size=(20, 4, 8),
        interpolatable=False,
    )
    shape = (labels_spec.roi / labels_spec.voxel_size).shape
    labels_data = np.ones(shape)
    labels_data[shape[0] // 2 :, :, :] += 2
    labels_data[:, shape[1] // 2 :, :] += 4
    labels_data[:, :, shape[2] // 2 :] += 8
    labels_array = Array(labels_data, labels_spec)

    labels_source = ArraySource(labels_key, labels_array)

    pipeline = labels_source + AddBoundaryDistanceGradients(
        label_array_key=labels_key,
        distance_array_key=dist_key,
        gradient_array_key=grad_key,
    )

    with build(pipeline):
        request = BatchRequest()
        request.add(labels_key, (120, 16, 64))
        request.add(dist_key, (120, 16, 64))
        request.add(grad_key, (120, 16, 64))

        batch = pipeline.request_batch(request)

        distances = batch.arrays[dist_key].data
        gradients = batch.arrays[grad_key].data
        shape = distances.shape

        g_001 = gradients[:, : shape[0] // 2, : shape[1] // 2, shape[2] // 2 :]
        g_101 = gradients[:, shape[0] // 2 :, : shape[1] // 2, shape[2] // 2 :]

        assert (g_001 == g_101).all()

        top = gradients[:, 0 : shape[0] // 2, :]
        bot = gradients[:, shape[0] : shape[0] // 2 - 1 : -1, :]

        assert (top == bot).all()
