import logging
import os
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder.volume import VolumeType
from gunpowder.points import PointsType, PointsOfType, SynPoint

logger = logging.getLogger(__name__)

class Snapshot(BatchFilter):
    '''Save a passing batch in an HDF file.'''

    def __init__(
            self,
            output_dir='snapshots',
            output_filename='{id}.hdf',
            every=1,
            additional_request=None):
        '''
        output_dir: string

            The directory to save the snapshots. Will be created, if it does not exist.

        output_filename: string

            Template for output filenames. '{id}' in the string will be replaced 
            with the ID of the batch. '{iteration}' with the training iteration 
            (if training was performed on this batch).

        every:

            How often to save a batch. 'every=1' indicates that every batch will 
            be stored, 'every=2' every second and so on. By default, every batch 
            will be stored.

        additional_request:

            An additional batch request to merge with the passing request, if a 
            snapshot is to be made. If not given, only the volumes that are in 
            the batch anyway are recorded.
        '''
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.every = max(1,every)
        self.additional_request = BatchRequest() if additional_request is None else additional_request
        self.n = 0

    def prepare(self, request):

        self.record_snapshot = self.n%self.every == 0
        self.n += 1

        # append additional volume requests, don't overwrite existing ones
        for volume_type, roi in self.additional_request.volumes.items():
            if volume_type not in request.volumes:
                request.volumes[volume_type] = roi

    def process(self, batch, request):

        if self.record_snapshot:

            try:
                os.makedirs(self.output_dir)
            except:
                pass

            snapshot_name = os.path.join(self.output_dir, self.output_filename.format(id=str(batch.id).zfill(8),iteration=batch.iteration))
            logging.info('saving to %s' %snapshot_name)
            with h5py.File(snapshot_name, 'w') as f:

                for (volume_type, volume) in batch.volumes.items():

                    ds_name = {
                            VolumeType.RAW: 'volumes/raw',
                            VolumeType.ALPHA_MASK: 'volumes/alpha_mask',
                            VolumeType.GT_LABELS: 'volumes/labels/neuron_ids',
                            VolumeType.GT_AFFINITIES: 'volumes/labels/affs',
                            VolumeType.GT_MASK: 'volumes/labels/mask',
                            VolumeType.GT_IGNORE: 'volumes/labels/ignore',
                            VolumeType.PRED_AFFINITIES: 'volumes/predicted_affs',
                            VolumeType.GT_BM_PRESYN: 'volumes/labels/gt_bm_presyn',
                            VolumeType.GT_BM_POSTSYN: 'volumes/labels/gt_bm_postsyn',
                            VolumeType.GT_MASK_EXCLUSIVEZONE_PRESYN: 'volumes/labels/gt_mask_exclusivezone_presyn',
                            VolumeType.GT_MASK_EXCLUSIVEZONE_POSTSYN: 'volumes/labels/gt_mask_exclusivezone_postsyn',
                            VolumeType.PRED_BM_PRESYN: 'volumes/predicted_bm_presyn',
                            VolumeType.PRED_BM_POSTSYN: 'volumes/predicted_bm_postsyn',
                    }[volume_type]

                    offset = volume.roi.get_offset()
                    offset*= volume.resolution
                    dataset = f.create_dataset(name=ds_name, data=volume.data)
                    dataset.attrs['offset'] = offset
                    dataset.attrs['resolution'] = volume.resolution

                if batch.loss is not None:
                    f['/'].attrs['loss'] = batch.loss


        self.n += 1

