import logging
import random
import numpy as np

# imports for deformed slice
from skimage.draw import line
from skimage.measure import label
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation

from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class DefectAugment(BatchFilter):

    def __init__(
            self,
            prob_missing=0.05,
            prob_low_contrast=0.05,
            prob_artifact=0.0,
            prib_deform=0.0,
            contrast_scale=0.1,
            artifact_source=None,
            deformation_strength=20,
            axis=0):
        '''Create a new DefectAugment node.

        Args

            prob_missing, prob_low_contrast, prob_artifact, prob_deform:

                Probabilities of having a missing section, low-contrast section,
                an artifact (see param 'artifact_source') or a deformed slice.
                The sum should not exceed 1.

        contrast_scale:

            By how much to scale the intensities for a low-contrast section.

        artifact_source:

            A gunpowder batch provider that delivers VolumeTypes.RAW and
            VolumeTypes.ALPHA_MASK, used if prob_artifact > 0.
                Strength of the deformation in slice.

        deformation_strength:

            Strength of the slice deformation.

        axis:

            Along which axis sections are cut.
        '''
        self.prob_missing = prob_missing
        self.prob_low_contrast = prob_low_contrast
        self.prob_artifact = prob_artifact
        self.prob_deform = prob_deform
        self.contrast_scale = contrast_scale
        self.artifact_source = artifact_source
        self.deformation_strength = deformation_strength
        self.axis = axis

    def setup(self):

        if self.artifact_source is not None:
            self.artifact_source.setup()

    def teardown(self):

        if self.artifact_source is not None:
            self.artifact_source.teardown()

    # send roi request to data-source upstream
    def prepare(self, request):

        # TODO ideally we would build the whole transformation here and then
        # check for the offsets to update the upstream ROI.
        # However, with the current logic of the defect augmentation trafos, this
        # is not trivial, hence we update the roi here if we have deformation trafos heuristically for now
        if self.prob_deform > 0.:
            spec = request[VolumeTypes.RAW]
            roi = spec.roi
            logger.debug("downstream request ROI is %s" % roi)

            # get roi in voxels
            roi /= self.voxel_size

            # create roi sufficiently large to feed deformation
            # TODO do we need to copy here?
            source_roi = roi
            growth = Coordinate((0, self.deformation_strength, self.deformation_strength))
            source_roi.grow(growth, growth)

            # update request ROI to get all voxels necessary to perfrom
            # transformation
            spec.roi = source_roi
            logger.debug("upstream request roi is %s" % spec.roi)

    def process(self, batch, request):

        assert batch.get_total_roi().dims() == 3, "defectaugment works on 3d batches only"

        prob_missing_threshold = self.prob_missing
        prob_low_contrast_threshold = prob_missing_threshold + self.prob_low_contrast
        prob_artifact_threshold = prob_low_contrast_threshold + self.prob_artifact
        prob_deform_slice = prob_artifact_threshold + self.prob_deform

        raw = batch.volumes[VolumeTypes.RAW]
        raw_voxel_size = self.spec[VolumeTypes.RAW].voxel_size

        for c in range((raw.spec.roi/raw_voxel_size).get_shape()[self.axis]):

            r = random.random()

            section_selector = tuple(
                slice(None if d != self.axis else c, None if d != self.axis else c+1)
                for d in range(raw.spec.roi.dims())
            )

            if r < prob_missing_threshold:

                logger.debug("Zero-out " + str(section_selector))
                raw.data[section_selector] = 0

            elif r < prob_low_contrast_threshold:

                logger.debug("Lower contrast " + str(section_selector))
                section = raw.data[section_selector]

                mean = section.mean()
                section -= mean
                section *= self.contrast_scale
                section += mean

                raw.data[section_selector] = section

            elif r < prob_artifact_threshold:

                logger.debug("Add artifact " + str(section_selector))
                section = raw.data[section_selector]

                alpha_voxel_size = self.artifact_source.spec[VolumeTypes.ALPHA_MASK].voxel_size

                assert raw_voxel_size == alpha_voxel_size, ("Can only alpha blend RAW with "
                                                            "ALPHA_MASK if both have the same "
                                                            "voxel size")

                artifact_request = BatchRequest()
                artifact_request.add(VolumeTypes.RAW, Coordinate(section.shape)*raw_voxel_size)
                artifact_request.add(VolumeTypes.ALPHA_MASK, Coordinate(section.shape)*alpha_voxel_size)
                logger.debug("Requesting artifact batch " + str(artifact_request))

                artifact_batch = self.artifact_source.request_batch(artifact_request)
                artifact_alpha = artifact_batch.volumes[VolumeTypes.ALPHA_MASK].data
                artifact_raw   = artifact_batch.volumes[VolumeTypes.RAW].data

                assert artifact_raw.dtype == section.dtype
                assert artifact_alpha.dtype == np.float32
                assert artifact_alpha.min() >= 0.0
                assert artifact_alpha.max() <= 1.0

                raw.data[section_selector] = section*(1.0 - artifact_alpha) + artifact_raw*artifact_alpha

            # TODO request bigger padded volume
            # => look into elastic augment for examples
            elif r < prob_deform_slice:

                logger.debug("Add deformed slice " + str(section_selector))
                section = raw.data[section_selector]
                shape = section.shape

                # randomly choose fixed x or fixed y with p = 1/2
                fixed_x = random.random() < .5
                if fixed_x:
                    x0, y0 = 0, np.random.randint(1, shape[1] - 2)
                    x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
                else:
                    x0, y0 = np.random.randint(1, shape[0] - 2), 0
                    x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

                ## generate the mask of the line that should be blacked out
                line_mask = np.zeros_like(section, dtype='bool')
                rr, cc = line(x0, y0, x1, y1)
                line_mask[rr, cc] = 1

                # generate vectorfield pointing towards the line to compress the image
                # first we get the unit vector representing the line
                line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
                line_vector /= np.linalg.norm(line_vector)
                # next, we generate the normal to the line
                normal_vector = np.zeros_like(line_vector)
                normal_vector[0] = - line_vector[1]
                normal_vector[1] = line_vector[0]

                # make meshgrid
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
                # generate the vector field
                flow_x, flow_y = np.zeros_like(section), np.zeros_like(section)

                # find the 2 components where coordinates are bigger / smaller than the line
                # to apply normal vector in the correct direction
                components = label(np.logical_not(line_mask).view('uint8'), background=0)
                assert len(np.unique(components)) == 3, "%i" % len(np.unique(components))
                neg_val = components[0, 0] if fixed_x else components[-1, -1]
                pos_val = components[-1, -1] if fixed_x else components[0, 0]

                flow_x[components == pos_val] = self.deformation_strength * normal_vector[1]
                flow_y[components == pos_val] = self.deformation_strength * normal_vector[0]
                flow_x[components == neg_val] = - self.deformation_strength * normal_vector[1]
                flow_y[components == neg_val] = - self.deformation_strength * normal_vector[0]

                # generate the flow fields
                flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)

                # set interpolation to cubic, spec interploatable is true, else to 0
                interpolation = 3 if self.spec[VolumeTypes.raw].interpolatable else 0

                # TODO no reflect once we have padding
                section = map_coordinates(
                    section, (flow_y, flow_x), mode='constant', order=interpolation
                ).reshape(shape)

                # dilate the line mask and zero out the section below it
                line_mask = binary_dilation(line_mask, iterations=10)
                section[line_mask] = 0.

                raw.data[section_selector] = section

            # in case we needed to change the ROI due to a deformation augment,
            # restore original ROI
            if self.prob_deform > 0.:
                raw.spec.roi = request[VolumeTypes.RAW].roi
