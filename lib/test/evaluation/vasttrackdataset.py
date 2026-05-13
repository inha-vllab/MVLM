import numpy as np
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str


class VastTrackDataset(BaseDataset):
    """
    VastTrack test set consisting of 3500 videos

    Publication:
        VastTrack: Vast Category Visual Object Tracking
        Liang Peng, Junyuan Gao, Xinran Liu, Weihong Li, Shaohua Dong, Zhipeng Zhang, Heng Fan and Libo Zhang
        https://arxiv.org/pdf/2403.03493.pdf

    Download the dataset from https://github.com/HengLan/VastTrack
    """

    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.vasttrack_dir, 'test')
        self.txt_file_path = os.path.join(os.path.dirname(__file__), "vasttrack_test_list.txt")
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls = '-'.join(self.sequence_list[i].split('.')[0].split('-')[:-1])
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = '-'.join(sequence_name.split('.')[0].split('-')[:-1])
        anno_path = os.path.join(self.base_path, class_name, sequence_name, 'Groundtruth.txt')

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        nlp_path = os.path.join(self.base_path, class_name, sequence_name, 'nlp.txt')
        nlp_rect = load_str(nlp_path)

        target_visible = ~((ground_truth_rect == [0, 0, 0, 0]).all(axis=1))

        frames_path = os.path.join(self.base_path, class_name, sequence_name, 'imgs')

        frames_list = [os.path.join(frames_path, '{:05d}.jpg'.format(frame_number)) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'vasttrack', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}'.format(self.txt_file_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list