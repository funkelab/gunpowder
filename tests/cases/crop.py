from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSourceCrop(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((200, 20, 20), (1800, 180, 180)),
                voxel_size=(20, 2, 2)))

        self.provides(
            PointsKeys.PRESYN,
            PointsSpec(
                roi=Roi((200, 20, 20), (1800, 180, 180))))

    def provide(self, request):
        pass

class TestCrop(ProviderTest):

    def test_output(self):

        cropped_roi_raw    = Roi((400, 40, 40), (1000, 100, 100))
        cropped_roi_presyn = Roi((800, 80, 80), (800, 80, 80))

        pipeline = (
            TestSourceCrop() +
            Crop(
                arrays = {ArrayKeys.RAW: cropped_roi_raw},
                points  = {PointsKeys.PRESYN: cropped_roi_presyn}))

        with build(pipeline):

            self.assertTrue(
                pipeline.spec[ArrayKeys.RAW].roi == cropped_roi_raw)
            self.assertTrue(
                pipeline.spec[PointsKeys.PRESYN].roi == cropped_roi_presyn)
