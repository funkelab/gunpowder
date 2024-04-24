import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    AsType,
    BatchRequest,
    MergeProvider,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_output():
    raw_key = ArrayKey("RAW")
    labels_key = ArrayKey("LABELS")
    raw_typed_key = ArrayKey("RAW_TYPECAST")
    labels_typed_key = ArrayKey("LABELS_TYPECAST")

    raw_spec = ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4))
    labels_spec = ArraySpec(
        roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)
    )

    roi = raw_spec.roi / raw_spec.voxel_size
    meshgrids = np.meshgrid(
        range(roi.get_begin()[0], roi.get_end()[0]),
        range(roi.get_begin()[1], roi.get_end()[1]),
        range(roi.get_begin()[2], roi.get_end()[2]),
        indexing="ij",
    )
    data = meshgrids[0] + meshgrids[1] + meshgrids[2]
    raw_array = Array(data, raw_spec)
    labels_array = Array(data, labels_spec)

    request = BatchRequest()
    request.add(raw_key, (200, 200, 200))
    request.add(raw_typed_key, (120, 120, 120))
    request.add(labels_key, (200, 200, 200))
    request.add(labels_typed_key, (200, 200, 200))

    pipeline = (
        (ArraySource(raw_key, raw_array), ArraySource(labels_key, labels_array))
        + MergeProvider()
        + AsType(raw_key, np.float16, raw_typed_key)
        + AsType(labels_key, np.int16, labels_typed_key)
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)

    for array_key, array in batch.arrays.items():
        # assert that pixels encode their position for supposedly unaltered
        # arrays
        if array_key in [raw_key, labels_key]:
            # the z,y,x coordinates of the ROI
            roi = array.spec.roi / 4
            meshgrids = np.meshgrid(
                range(roi.get_begin()[0], roi.get_end()[0]),
                range(roi.get_begin()[1], roi.get_end()[1]),
                range(roi.get_begin()[2], roi.get_end()[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            assert np.array_equal(array.data, data)

        elif array_key == raw_typed_key:
            assert array.data.dtype == np.float16
            assert int(array.data[1, 11, 1]) == 43

        elif array_key == labels_typed_key:
            assert array.data.dtype == np.int16
            assert int(array.data[1, 11, 1]) == 13
