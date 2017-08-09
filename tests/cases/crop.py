from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSourceCrop(BatchProvider):

    def __init__(self):
        self.spec = ProviderSpec()

    def setup(self):
        self.spec.volumes[VolumeTypes.RAW]   = Roi((10, 10, 10), (90, 90, 90))
        self.spec.points[PointsTypes.PRESYN] = Roi((20, 20, 20), (70, 70, 70))

    def get_spec(self):
        return self.spec

    def provide(self, request):

        batch = Batch()

        if VolumeTypes.RAW in request.volumes:
            raw_roi_specs   = self.get_spec().volumes[VolumeTypes.RAW]
            raw_roi_request = request.volumes[VolumeTypes.RAW]

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(raw_roi_specs.get_end()[0]),
                    range(raw_roi_specs.get_end()[1]),
                    range(raw_roi_specs.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            shape_request  = raw_roi_request.get_shape()
            offset_request = raw_roi_request.get_offset()
            data = data[offset_request[0]:offset_request[0] + shape_request[0],
                        offset_request[1]:offset_request[1] + shape_request[1],
                        offset_request[2]:offset_request[2] + shape_request[2]]

            batch.volumes[VolumeTypes.RAW] = Volume(data, raw_roi_request)


        if PointsTypes.PRESYN in request.points:

            all_locations = {
                                1: np.asarray([20, 20, 20]),
                                2: np.asarray([40, 40, 40]),
                                3: np.asarray([50, 50, 50]),
                                4: np.asarray([70, 70, 70]),
                                5: np.asarray([80, 80, 80]),
                                6: np.asarray([90, 90, 90]),
            }

            data_points = {}
            for loc_id, loc in all_locations.items():
                if request.points[PointsTypes.PRESYN].contains(Coordinate(loc)):
                    data_points[loc_id] = Point(location=loc)

            batch.points[PointsTypes.PRESYN] = Points(data=data_points,
                                                      roi = request.points[PointsTypes.PRESYN],
                                                      resolution=(2,2,2))

        return batch


class TestCrop(ProviderTest):

    def test_output(self):

        cropped_roi_raw    = Roi((20, 20, 20), (50, 50, 50))
        cropped_roi_presyn = Roi((40, 40, 40), (40, 40, 40))

        pipeline = TestSourceCrop() + Crop(
                                           volumes = {VolumeTypes.RAW: cropped_roi_raw},
                                           points  = {PointsTypes.PRESYN: cropped_roi_presyn}
                                           )

        pipeline_with_randomlocation = TestSourceCrop() + Crop(
                                                               volumes = {VolumeTypes.RAW: cropped_roi_raw},
                                                               points  = {PointsTypes.PRESYN: cropped_roi_presyn}
                                                               ) \
                                                        + RandomLocation()

        with build(pipeline) as p:

            request = BatchRequest()
            request.add_volume_request(VolumeTypes.RAW, cropped_roi_raw.get_shape())
            request.add_points_request(PointsTypes.PRESYN, cropped_roi_presyn.get_shape())

            request.volumes[VolumeTypes.RAW]   += cropped_roi_raw.get_offset()
            request.points[PointsTypes.PRESYN] += cropped_roi_raw.get_offset()

            batch = p.request_batch(request)

            # specs of pipeline are specs defined by user in crop
            self.assertTrue(pipeline.get_spec().volumes[VolumeTypes.RAW]   == cropped_roi_raw)
            self.assertTrue(pipeline.get_spec().points[PointsTypes.PRESYN] == cropped_roi_presyn)

            # all point locations lie within roi defined by user in crop
            for point_id, point in batch.points[PointsTypes.PRESYN].data.items():
                self.assertTrue(cropped_roi_presyn.contains(Coordinate(point.location)))

            raw_roi_specs = Roi((10, 10, 10), (90, 90, 90))
            raw_roi_request = request.volumes[VolumeTypes.RAW]

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(raw_roi_specs.get_end()[0]),
                range(raw_roi_specs.get_end()[1]),
                range(raw_roi_specs.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            shape_request = raw_roi_request.get_shape()
            offset_request = raw_roi_request.get_offset()
            data = data[offset_request[0]:offset_request[0] + shape_request[0],
                   offset_request[1]:offset_request[1] + shape_request[1],
                   offset_request[2]:offset_request[2] + shape_request[2]]

            # correct part of data was returned
            self.assertTrue((data == batch.volumes[VolumeTypes.RAW].data).all())


        with build(pipeline_with_randomlocation) as p_rl:

            common_roi = cropped_roi_raw.intersect(cropped_roi_presyn)

            request_shape = Coordinate(common_roi.get_shape() + (10,10,10))
            request = BatchRequest()
            request.add_volume_request(VolumeTypes.RAW, request_shape)
            request.add_points_request(PointsTypes.PRESYN, request_shape)

            with self.assertRaises(AttributeError):
                batch = p_rl.request_batch(request)

            request_shape = Coordinate(common_roi.get_shape()-(1,1,1))
            request = BatchRequest()
            request.add_volume_request(VolumeTypes.RAW, request_shape)
            request.add_points_request(PointsTypes.PRESYN, request_shape)
            batch = p_rl.request_batch(request)

            self.assertTrue(batch.volumes[VolumeTypes.RAW].data.shape == request_shape)









