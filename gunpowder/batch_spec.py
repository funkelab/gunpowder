import multiprocessing

class BatchSpec:
    '''A possibly partial specification of a batch.
    '''

    next_id = multiprocessing.Value('L')

    @staticmethod
    def get_next_id():
        with BatchSpec.next_id.get_lock():
            next_id = BatchSpec.next_id.value
            BatchSpec.next_id.value += 1
        return next_id

    def __init__(self, shape, offset=None, resolution=None, with_gt=False, with_gt_mask=False):
        self.shape = shape
        self.offset = offset
        self.resolution = resolution
        self.with_gt = with_gt
        self.with_gt_mask = with_gt_mask
        self.id = BatchSpec.get_next_id()
        print("BatchSpec: created new spec with id " + str(self.id))

    def get_bounding_box(self):

        if self.offset is None:
            return None

        return tuple(
                slice(self.offset[d], self.shape[d] + self.offset[d])
                for d in range(len(self.shape))
        )
