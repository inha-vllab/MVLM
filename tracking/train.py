import os
import sys
import subprocess
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--launcher', type=str, choices=["launch", "torchrun"], default="launch",
                        help="distributed launcher for multiple mode: "
                             "'launch' uses python -m torch.distributed.launch (default), "
                             "'torchrun' uses torchrun")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    run_script = os.path.join('lib', 'train', 'run_training.py')
    if args.mode == "single":
        train_cmd = [sys.executable, run_script,
                     '--script', args.script, '--config', args.config, '--save_dir', args.save_dir]
    elif args.mode == "multiple":
        if args.launcher == "torchrun":
            train_cmd = ['torchrun', '--nproc_per_node', str(args.nproc_per_node), run_script,
                         '--script', args.script, '--config', args.config, '--save_dir', args.save_dir]
        else:
            train_cmd = [sys.executable, '-m', 'torch.distributed.launch',
                         '--nproc_per_node', str(args.nproc_per_node), run_script,
                         '--script', args.script, '--config', args.config, '--save_dir', args.save_dir]
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    print(' '.join(train_cmd))
    subprocess.run(train_cmd)


if __name__ == "__main__":
    main()
