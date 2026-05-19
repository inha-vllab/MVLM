import os
import sys
import argparse
import time
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

def run_tracker(args):
    tracker_name = args.tracker_name
    tracker_param = args.tracker_param
    exp_id = args.exp_id
    dataset_name = args.dataset_name
    weight_path = args.weight_path
    sequence = args.sequence
    debug = args.debug
    threads = args.threads
    num_gpus = args.num_gpus
    parallel_mode = args.mode

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, exp_id, dataset_name, weight_path)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, parallel_mode=parallel_mode)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--exp_id', type=str, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (lasot, tnl2k, otb99_lang, mgit).')
    parser.add_argument('--weight_path', type=str, default=None, help='The weight path.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', action='store_true', help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'dist', 'mp'],
                        help="Execution mode: "
                             "'single' runs sequentially on 1 GPU (default), "
                             "'dist' for torchrun/torch.distributed (must launch via torchrun), "
                             "'mp' for multiprocessing.Pool via plain python.")

    args = parser.parse_args()

    try:
        args.sequence = int(args.sequence)
    except:
        pass

    start_time = time.time()
    run_tracker(args)
    end_time = time.time()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        print(f"Time taken: {end_time - start_time:.1f} seconds")

if __name__ == '__main__':
    main()
