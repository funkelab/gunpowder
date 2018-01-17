from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
import socket

class TestDvidSource(ProviderTest):

    def test_output_3d(self):

        dvid_server = 'slowpoke1'

        # check if DVID server is available:
        try:
            socket.gethostbyname(dvid_server)
        except:
            print("DVID server not available, skipping TestDvidSource")
            return

        # create array keys
        raw = ArrayKey('RAW')
        seg = ArrayKey('SEG')
        mask = ArrayKey('MASK')

        pipeline = (
            DvidSource(
                dvid_server,
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
                output_filename = 'dvid_source_test.hdf'
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
