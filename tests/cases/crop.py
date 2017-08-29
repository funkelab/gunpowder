from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSourceCrop(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.RAW,
            VolumeSpec(
                roi=Roi((200, 20, 20), (1800, 180, 180)),
                voxel_size=(20, 2, 2)))

        self.provides(
            PointsTypes.PRESYN,
            PointsSpec(
                roi=Roi((200, 20, 20), (1800, 180, 180))))

    def provide(self, request):

        batch = Batch()

        if VolumeTypes.RAW in request:

            raw_roi_specs   = self.spec[VolumeTypes.RAW].roi
            raw_roi_request = request[VolumeTypes.RAW].roi
            raw_voxel_size  = self.spec[VolumeTypes.RAW].voxel_size
            # the z,y,x coordinates of the ROI

            meshgrids = np.meshgrid(
                    range(raw_roi_specs.get_end()[0]//raw_voxel_size[0]),
                    range(raw_roi_specs.get_end()[1]//raw_voxel_size[1]),
                    range(raw_roi_specs.get_end()[2]//raw_voxel_size[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            shape_request  = raw_roi_request.get_shape() // raw_voxel_size
            offset_request = raw_roi_request.get_offset()  // raw_voxel_size
            data = data[offset_request[0]:offset_request[0] + shape_request[0],
                        offset_request[1]:offset_request[1] + shape_request[1],
                        offset_request[2]:offset_request[2] + shape_request[2]]

            spec = self.spec[VolumeTypes.RAW].copy()
            spec.roi = raw_roi_request
            batch.volumes[VolumeTypes.RAW] = Volume(data, spec)


        if PointsTypes.PRESYN in request:

            all_locations = {
                                1: np.asarray([400, 40, 40]),
                                2: np.asarray([800, 80, 80]),
                                3: np.asarray([1000, 100, 100]),
                                4: np.asarray([1400, 140, 140]),
                                5: np.asarray([1600, 160, 160]),
                                6: np.asarray([1800, 180, 180]),
            }

            data_points = {}
            for loc_id, loc in all_locations.items():
                if request[PointsTypes.PRESYN].roi.contains(Coordinate(loc)):
                    data_points[loc_id] = Point(location=loc)

            batch.points[PointsTypes.PRESYN] = Points(data=data_points,
                                                      spec=PointsSpec(roi=request[PointsTypes.PRESYN].roi))

        return batch


class TestCrop(ProviderTest):

    def test_output(self):

        cropped_roi_raw    = Roi((400, 40, 40), (1000, 100, 100))
        cropped_roi_presyn = Roi((800, 80, 80), (800, 80, 80))

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
            request.add(VolumeTypes.RAW, cropped_roi_raw.get_shape())
            request.add(PointsTypes.PRESYN, cropped_roi_presyn.get_shape())
            # move request to provided crop rois
            request[VolumeTypes.RAW].roi.set_offset(cropped_roi_raw.get_offset())
            request[PointsTypes.PRESYN].roi.set_offset(cropped_roi_presyn.get_offset())

            batch = p.request_batch(request)

            # specs of pipeline are specs defined by user in crop
            self.assertTrue(pipeline.spec[VolumeTypes.RAW].roi   == cropped_roi_raw)
            self.assertTrue(pipeline.spec[PointsTypes.PRESYN].roi == cropped_roi_presyn)

            # all point locations lie within roi defined by user in crop
            for point_id, point in batch.points[PointsTypes.PRESYN].data.items():
                self.assertTrue(cropped_roi_presyn.contains(Coordinate(point.location)))

            raw_roi_specs = Roi((200, 20, 20), (1800, 180, 180))
            raw_roi_request = request[VolumeTypes.RAW].roi
            raw_voxel_size  = pipeline.spec[VolumeTypes.RAW].voxel_size

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(raw_roi_specs.get_end()[0]//raw_voxel_size[0]),
                range(raw_roi_specs.get_end()[1]//raw_voxel_size[1]),
                range(raw_roi_specs.get_end()[2]//raw_voxel_size[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            shape_request = raw_roi_request.get_shape() // raw_voxel_size
            offset_request = raw_roi_request.get_offset()  // raw_voxel_size
            data = data[offset_request[0]:offset_request[0] + shape_request[0],
                   offset_request[1]:offset_request[1] + shape_request[1],
                   offset_request[2]:offset_request[2] + shape_request[2]]

            # correct part of data was returned
            self.assertTrue((data == batch.volumes[VolumeTypes.RAW].data).all())


        with build(pipeline_with_randomlocation) as p_rl:

            common_roi = cropped_roi_raw.intersect(cropped_roi_presyn)

            request_shape = Coordinate(common_roi.get_shape() + (200,20,20))
            request = BatchRequest()
            request.add(VolumeTypes.RAW, request_shape)
            request.add(PointsTypes.PRESYN, request_shape)

            with self.assertRaises(AttributeError):
                batch = p_rl.request_batch(request)

            request_shape = Coordinate(common_roi.get_shape()-(20, 2, 2))  # (1,1,1))
            request = BatchRequest()
            request.add(VolumeTypes.RAW, request_shape)
            request.add(PointsTypes.PRESYN, request_shape)
            batch = p_rl.request_batch(request)

            self.assertTrue(batch.volumes[VolumeTypes.RAW].data.shape == request_shape//raw_voxel_size)
