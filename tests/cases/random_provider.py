import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    RandomProvider,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_output():
    a = ArrayKey("A")
    source_a = ArraySource(
        a,
        Array(
            np.zeros((5, 15, 25), dtype=np.uint8),
            spec=ArraySpec(
                roi=Roi((0, 0, 0), (50, 60, 50)), voxel_size=(10, 4, 2), dtype=np.uint8
            ),
        ),
    )
    source_b = ArraySource(
        a,
        Array(
            np.ones((5, 15, 25), dtype=np.uint8),
            spec=ArraySpec(
                roi=Roi((0, 0, 0), (50, 60, 50)), voxel_size=(10, 4, 2), dtype=np.uint8
            ),
        ),
    )
    random_provider = ArrayKey("RANDOM_PROVIDER")

    pipeline = (source_a, source_b) + RandomProvider(
        random_provider_key=random_provider
    )

    with build(pipeline):
        possibilities = set([0, 1])
        seen = set()
        for i in range(10):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        random_provider: ArraySpec(nonspatial=True),
                    }
                )
            )

            value = batch.arrays[a].data[0, 0, 0]
            assert value in possibilities
            assert batch.arrays[random_provider].data.item() == value
            seen.add(value)
            if len(possibilities - seen) == 0:
                break

        assert seen == possibilities
