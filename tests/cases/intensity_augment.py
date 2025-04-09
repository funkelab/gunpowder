import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    IntensityAugment,
    Roi,
    build,
)

from .helper_sources import ArraySource


@pytest.mark.parametrize("slab", [None, (1, -1, -1)])
@pytest.mark.parametrize("z_section_wise", [None, True])
def test_shift(slab, z_section_wise):
    raw_key = ArrayKey("RAW")
    raw_spec = ArraySpec(
        roi=Roi((0, 0, 0), (10, 10, 10)), voxel_size=(1, 1, 1), dtype=np.float32
    )
    raw_data = np.random.randn(*(raw_spec.roi.shape / raw_spec.voxel_size)).astype(
        np.float32
    )
    raw_array = Array(raw_data, raw_spec)

    if z_section_wise is not None and slab is not None:
        with pytest.raises(AssertionError):
            pipeline = ArraySource(raw_key, raw_array) + IntensityAugment(
                raw_key,
                scale_min=0,
                scale_max=0,
                shift_min=0.5,
                shift_max=0.5,
                clip=False,
                z_section_wise=z_section_wise,
                slab=slab,
            )
        return
    else:
        pipeline = ArraySource(raw_key, raw_array) + IntensityAugment(
            raw_key,
            scale_min=0,
            scale_max=0,
            shift_min=0.5,
            shift_max=0.5,
            clip=False,
            z_section_wise=z_section_wise,
            slab=slab,
        )

    request = BatchRequest()
    request.add(raw_key, (10, 10, 10))

    with build(pipeline):
        batch = pipeline.request_batch(request)

        x = batch.arrays[raw_key].data

        # subtract mean of unshifted data since intensity augment
        # scales intensity from the mean
        if z_section_wise is not None or slab is not None:
            x -= np.mean(raw_data, axis=(1, 2), keepdims=True)
        else:
            x -= np.mean(raw_data)
        assert np.isclose(x.min(), 0.5)
        assert np.isclose(x.max(), 0.5)
