from batch_filter import BatchFilter

import logging
logger = logging.getLogger(__name__)

class CropGt(BatchFilter):
    '''Crop the ground-truth in a batch to the networks output size as given in 
    the batch spec.

    Optionally, you can specify an additional padding to leave around the 
    requested output size (but remember that ultimately you will have to crop to 
    the output size before passing the batch into the training).
    '''

    def __init__(self, additional_padding=0):
        self.additional_padding = additional_padding

    def process(self, batch):

        input_shape = batch.spec.shape
        output_shape = batch.spec.output_shape
        padded_output_shape = tuple(
                output_shape[d]+self.additional_padding*2
                for d in range(len(output_shape))
        )

        assert batch.gt is not None or batch.gt_affinities is not None, "You are trying to crop the ground-truth in a batch that doesn't have one."

        for d in range(len(input_shape)):
            assert (input_shape[d]-output_shape[d])%2==0, "Output shape in dimension %d (%d) can not be centered in input (%d), I don't know what to do."%(d,output_shape[d],input_shape[d])
            assert padded_output_shape[d] <= input_shape[d], "Your requested output shape is larger than the input shape, I don't know what to do."

        offset = tuple(
                (input_shape[d]-padded_output_shape[d])/2
                for d in range(len(input_shape))
        )

        # if gt was cropped before, we can't use 'offset' above to crop in the 
        # array
        if batch.gt_offset is not None:
            offset_in_gt = tuple(
                    offset[d] - batch.gt_offset[d]
                    for d in range(len(input_shape))
            )
        else:
            offset_in_gt = offset

        if batch.gt is not None:
            batch.gt = self.__crop(batch.gt, offset_in_gt, padded_output_shape)

        if batch.gt_mask is not None:
            batch.gt_mask = self.__crop(batch.gt_mask, offset_in_gt, padded_output_shape)

        if batch.gt_affinities is not None:
            batch.gt_affinities = self.__crop(batch.gt_affinities, (0,) + offset_in_gt, (len(batch.gt_affinities),) + padded_output_shape)

        batch.gt_offset = offset

    def __crop(self, a, offset, shape):

        logger.debug("Cropping GT from %s to %s"%(str(a.shape), shape))

        for d in range(len(shape)):
            assert a.shape[d] >= shape[d], "The ground-truth was already cropped to a smaller size, I don't know what to do."

        crop = tuple(
                slice(offset[d], offset[d]+shape[d])
                for d in range(len(shape))
        )

        return a[crop]


