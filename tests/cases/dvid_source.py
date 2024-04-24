import logging
import socket

import pytest

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    DvidSource,
    Roi,
    Snapshot,
    build,
)
from gunpowder.ext import NoSuchModule, dvision

logger = logging.getLogger(__name__)


DVID_SERVER = "slowpoke1"


def is_dvid_unavailable(server):
    if isinstance(dvision, NoSuchModule):
        return True
    try:
        socket.gethostbyname(server)
        return False
    except Exception:  # todo: make more specific
        return True


@pytest.mark.skipif(
    is_dvid_unavailable(DVID_SERVER), reason="DVID server not available"
)
def test_output_3d(tmpdir):
    # create array keys
    raw = ArrayKey("RAW")
    seg = ArrayKey("SEG")
    mask = ArrayKey("MASK")

    pipeline = DvidSource(
        DVID_SERVER,
        32768,
        "2ad1d8f0f172425c9f87b60fd97331e6",
        datasets={raw: "grayscale", seg: "groundtruth"},
        masks={mask: "seven_column"},
    ) + Snapshot(
        {
            raw: "/volumes/raw",
            seg: "/volumes/labels/neuron_ids",
            mask: "/volumes/labels/mask",
        },
        output_dir=tmpdir,
        output_filename="dvid_source_test{id}-{iteration}.hdf",
    )

    with build(pipeline):
        batch = pipeline.request_batch(
            BatchRequest(
                {
                    raw: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80))),
                    seg: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80))),
                    mask: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80))),
                }
            )
        )

        assert batch.arrays[raw].spec.interpolatable
        assert not batch.arrays[seg].spec.interpolatable
        assert not batch.arrays[mask].spec.interpolatable

        assert batch.arrays[raw].spec.voxel_size == (8, 8, 8)
        assert batch.arrays[seg].spec.voxel_size == (8, 8, 8)
        assert batch.arrays[mask].spec.voxel_size == (8, 8, 8)
