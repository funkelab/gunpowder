from gunpowder.batch import Batch
import numpy as np
from gunpowder import WKWSource, pipeline
from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.batch_request import BatchRequest
from gunpowder.build import build
from gunpowder import Roi
from webknossos import Dataset


def _create_dataset(
    data_file, key, data,
    magnification=(1, 1, 1), category='color', **kwargs
):
    Dataset.get_or_create(data_file, scale=(1, 1, 1))\
        .get_or_add_layer(key, category, dtype_per_layer=data.dtype, **kwargs)\
        .get_or_add_mag(magnification)\
        .write(data)

def test_WKKSource__data_integrity(tmp_path):
    wkw_file = tmp_path / 'data_integrity'

    data = np.arange(27,dtype=np.uint8).reshape(3,3,3)


    Dataset.get_or_create(wkw_file, scale=(1,2,3))\
        .get_or_add_layer('seg', 'segmentation', largest_segment_id=int(data.max()))\
        .get_or_add_mag(1)\
        .write(data)

    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file,
        {seg: 'seg'},
        mag_specs = {seg: 1}
    )

    request = BatchRequest()
    request[seg] = Roi((0, 0, 0), (3,6,9))

    with build(source):
        batch = source.request_batch(request)

    assert np.all(batch[seg].data == data)

def test_WKWSource__data_integrity__offset(tmp_path):
    wkw_file = tmp_path / 'data_integrity_offset'

    data = np.arange(5*5*5, dtype=np.uint8).reshape(5,5,5)

    Dataset.get_or_create(wkw_file, scale=(1,2,3))\
        .get_or_add_layer('seg', 'segmentation', largest_segment_id=int(data.max()))\
        .get_or_add_mag(1)\
        .write(data)

    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file, 
        {seg: 'seg'},
        mag_specs = {seg: 1} 
    )

    request = BatchRequest()
    request[seg] = Roi((3, 2, 0), (2, 4, 9))

    with build(source):
        batch = source.request_batch(request)

    assert np.all(batch[seg].data == data[3:, 1:3, :3])


def test_WKWSource(tmp_path):

    wkw_file = tmp_path / 'wkw_test'

    _create_dataset(
        wkw_file, 'raw', np.zeros((100, 100, 100), dtype=np.float32))
    _create_dataset(
        wkw_file, 'raw_low', np.zeros((16, 16, 16), dtype=np.float32),
        magnification=(16, 16, 16))
    _create_dataset(
        wkw_file, 'seg', np.ones((100, 100, 100), dtype=np.uint64),
        category='segmentation', largest_segment_id=1)

    raw = ArrayKey('RAW')
    raw_low = ArrayKey('RAW_LOW')
    seg = ArrayKey('SEG')

    source = WKWSource(
        wkw_file,
        {
            raw: 'raw',
            raw_low: 'raw_low',
            seg: 'seg'
        },
        mag_specs={
            raw: 1,
            raw_low: [16, 16, 16],
            seg: [1, 1, 1],
        }
    )

    with build(source):
        batch = source.request_batch(
            BatchRequest({
                raw: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                raw_low: ArraySpec(roi=Roi((0, 0, 0), (128, 128, 128))),
                seg: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
            })
        )

        assert batch.arrays[raw].spec.interpolatable
        assert batch.arrays[raw_low].spec.interpolatable
        assert not batch.arrays[seg].spec.interpolatable
