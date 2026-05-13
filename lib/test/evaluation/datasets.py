from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    vasttrack=DatasetInfo(module=pt % "vasttrack", class_name="VastTrackDataset", kwargs=dict()),
    otb99_lang=DatasetInfo(module=pt % "otb99lang", class_name="OTB99LangDataset", kwargs=dict()),
    mgit=DatasetInfo(module=pt % "videocube", class_name="VideoCubeDataset", kwargs=dict(split='test')),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
