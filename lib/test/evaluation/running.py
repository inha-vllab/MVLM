import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch
import torch.distributed as dist


len_dataset = 0

def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['lasot', 'tnl2k', 'vasttrack', 'otb99_lang', 'mgit']:
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset), exist_ok=True)
    '''2021.1.5 create new folder for these three datasets'''
    if seq.dataset in ['lasot', 'tnl2k', 'vasttrack', 'otb99_lang', 'mgit']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)
            

def run_sequence(seq: Sequence, tracker: Tracker, debug=False, len_dataset=0, num_gpu=8):
    """Runs a tracker on a sequence."""
    # Set GPU based on mp worker ID (no-op in dist mode where device is set externally)
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        torch.cuda.set_device(worker_id % num_gpu)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['lasot', 'tnl2k', 'vasttrack', 'otb99_lang', 'mgit']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = os.path.join(tracker.results_dir, seq.name + '.txt')
            return os.path.isfile(bbox_file)
        else:
            bbox_files = [os.path.join(tracker.results_dir, '{}_{}.txt'.format(seq.name, obj_id)) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0
    
    if _results_exist():
        print('FPS: {}'.format(-1))
        return

    print('Start running sequence: {}'.format(seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    _save_tracker_output(seq, tracker, output)
        
    def _num_results_done():
        base_results_path = os.path.join(tracker.results_dir, seq.dataset)
        done_seqs = [f for f in os.listdir(base_results_path) if f.endswith('.txt')]
        return int(len(done_seqs) / 2)

    print(f'FPS: {num_frames / exec_time:.2f}  /  done seqs: {_num_results_done()}/{len_dataset} ({_num_results_done()/len_dataset*100:.1f}%)')


def _init_dist(num_gpus):
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="gloo" if os.name == "nt" else "nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    torch.cuda.set_device(local_rank % num_gpus)
    return local_rank, (dist.get_world_size() if dist.is_initialized() else 1)

def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=None, parallel_mode='single'):
    """Runs a list of trackers on a dataset.

    args:
        parallel_mode: 'single' runs sequentially on 1 GPU (default);
                       'dist' uses torchrun/torch.distributed;
                       'mp' uses multiprocessing.Pool launched from plain python.
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if parallel_mode == 'single':
        len_dataset = len(dataset)
        print(f'Evaluating {len(trackers)} tracker(s) on {len_dataset} sequences [single GPU mode]')
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, len_dataset=len_dataset, num_gpu=1)
        print("Done")

    elif parallel_mode == 'dist':
        rank, world = _init_dist(num_gpus)
        dataset_split = dataset[rank::world]
        len_dataset = len(dataset)

        print(f"[RANK {rank}] GPU cuda:{rank} seqs={len(dataset_split)}/{len_dataset}", flush=True)

        for seq in dataset_split:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, len_dataset=len_dataset)

        if rank == 0:
            print("Done")

    elif parallel_mode == 'mp':
        multiprocessing.set_start_method('spawn', force=True)
        len_dataset = len(dataset)
        print(f'Evaluating {len(trackers)} tracker(s) on {len_dataset} sequences '
              f'[mp mode, threads={threads}, num_gpus={num_gpus}]')

        if threads <= 1:
            for seq in dataset:
                for tracker_info in trackers:
                    run_sequence(seq, tracker_info, debug=debug, len_dataset=len_dataset, num_gpu=num_gpus)
        else:
            param_list = [
                (seq, tracker_info, debug, len_dataset, num_gpus)
                for seq, tracker_info in product(dataset, trackers)
            ]
            with multiprocessing.Pool(processes=threads) as pool:
                pool.starmap(run_sequence, param_list)

        print("Done")

    else:
        raise ValueError(f"Unknown parallel_mode '{parallel_mode}'. Choose 'single', 'dist', or 'mp'.")