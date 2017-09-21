import logging

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBlobsFromPoints(BatchFilter):

    def __init__(self, blob_settings):

        self.blob_settings = blob_settings


    def setup(self):
        for blob_name, settings in self.blob_settings.items():
            self.provides(
                settings['output_volume_type'],
                self.spec[settings['restrictive_mask_type']]
                )

    def prepare(self, request):

        for blob_name, settings in self.blob_settings.items():
            volume_type = settings['output_volume_type']
            if volume_type in request:

                point_type = settings['point_type']

                # If point is not already requested, add to request
                if point_type not in request.points_specs:
                    request.add(point_type, request[volume_type].roi.get_shape())

                # this node will provide this volume type
                del request[volume_type]
            else:
                # do nothing if no blobs of this type were requested
                logger.warning('{} output volume type for {} never requested. \
                    Deleting entry...'.format(settings['output_volume_type'], blob_name))
                del self.blob_settings[blob_name]


    def process(self, batch, request):

        for blob_name, settings in self.blob_settings.items():
            point_type = settings['point_type']
            volume_type = settings['output_volume_type']
            voxel_size = settings['output_voxel_size']
            restrictive_mask_type = settings['restrictive_mask_type']

            assert point_type in batch.points, "Upstream does not provide required point type\
            : {}".format(point_type)

            assert restrictive_mask_type in batch.volumes, "Upstream does not provide required \
            volume type: {}".format(restrictive_mask_type)
