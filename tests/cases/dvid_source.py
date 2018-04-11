from .provider_test import ProviderTest
from unittest import skipIf
from gunpowder import *
import numpy as np
import socket
import logging

logger = logging.getLogger(__name__)


DVID_SERVER = 'slowpoke1'


def is_dvid_unavailable(server):
    try:
        socket.gethostbyname(server)
        return False
    except Exception:  # todo: make more specific
        return True


class TestDvidSource(ProviderTest):

    @skipIf(is_dvid_unavailable(DVID_SERVER), 'DVID server not available')
    def test_output_3d(self):

        # create array keys
        raw = ArrayKey('RAW')
        seg = ArrayKey('SEG')
        mask = ArrayKey('MASK')

        pipeline = (
            DvidSource(
                DVID_SERVER,
                32768,
                '2ad1d8f0f172425c9f87b60fd97331e6',
                datasets = {
                    raw: 'grayscale',
                    seg: 'groundtruth'
                },
                masks = {
                    mask: 'seven_column'
                }
            ) +
            Snapshot(
                {
                    raw: '/volumes/raw',
                    seg: '/volumes/labels/neuron_ids',
                    mask: '/volumes/labels/mask'
                },
                output_dir=self.path_to(),
                output_filename='dvid_source_test{id}-{iteration}.hdf'
            )
        )

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest({
                    raw: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80))),
                    seg: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80))),
                    mask: ArraySpec(roi=Roi((33000, 15000, 20000), (32000, 8, 80)))
                })
            )

            self.assertTrue(batch.arrays[raw].spec.interpolatable)
            self.assertFalse(batch.arrays[seg].spec.interpolatable)
            self.assertFalse(batch.arrays[mask].spec.interpolatable)

            self.assertEqual(batch.arrays[raw].spec.voxel_size, (8, 8, 8))
            self.assertEqual(batch.arrays[seg].spec.voxel_size, (8, 8, 8))
            self.assertEqual(batch.arrays[mask].spec.voxel_size, (8, 8, 8))
