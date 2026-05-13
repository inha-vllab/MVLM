import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tracker_name", type=str, default='mvlm')
parser.add_argument("--tracker_param", type=str, default='mvlm_b224')
parser.add_argument("--exp_id", type=str, default='')
parser.add_argument("--dataset", type=str, default='tnl2k')
args = parser.parse_args()

trackers = []
dataset_name = args.dataset
# choosen from 'lasot', 'tnl2k', 'vasttrack', 'otb99_lang', 'mgit'

trackers.append(Tracker(name=args.tracker_name, parameter_name=args.tracker_param, exp_id=args.exp_id, dataset_name=dataset_name, weight_path='', display_name=args.tracker_param))

dataset = get_dataset(dataset_name)

report_text = print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)