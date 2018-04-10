from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
import logging

logger = logging.getLogger(__name__)


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

        PointsKey('PRESYN')

        pipeline = (
            TestSourceCrop() +
            Crop(ArrayKeys.RAW, cropped_roi_raw) +
            Crop(PointsKeys.PRESYN, cropped_roi_presyn))

        with build(pipeline):

            self.assertTrue(
                pipeline.spec[ArrayKeys.RAW].roi == cropped_roi_raw)
            self.assertTrue(
                pipeline.spec[PointsKeys.PRESYN].roi == cropped_roi_presyn)

        pipeline = (
            TestSourceCrop() +
            Crop(
                ArrayKeys.RAW,
                fraction_negative=(0.25, 0, 0),
                fraction_positive=(0.25, 0, 0)))
        expected_roi_raw = Roi((650, 20, 20), (900, 180, 180))

        with build(pipeline):

            logger.info(pipeline.spec[ArrayKeys.RAW].roi)
            logger.info(expected_roi_raw)
            self.assertTrue(
                pipeline.spec[ArrayKeys.RAW].roi == expected_roi_raw)
