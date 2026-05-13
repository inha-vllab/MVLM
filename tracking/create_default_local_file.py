import argparse
import os


def _fwd(path):
    """Normalize path to forward slashes for safe embedding in Python source."""
    return path.replace('\\', '/')


def create_local_config(workspace_dir, data_dir, save_dir):
    """Generate lib/local_config.py with dataset and output paths."""
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(prj_dir, 'lib', 'local_config.py')

    lines = [
        'class EnvSettings:\n',
        '    def __init__(self):\n',
        '        # ── Workspace & output ─────────────────────────────────────────────\n',
        f"        self.workspace_dir       = '{_fwd(workspace_dir)}'\n",
        f"        self.tensorboard_dir     = '{_fwd(os.path.join(workspace_dir, 'tensorboard'))}'\n",
        f"        self.save_dir            = '{_fwd(os.path.join(save_dir))}'\n",
        f"        self.results_dir         = '{_fwd(os.path.join(save_dir, 'test', 'tracking_results'))}'\n",
        f"        self.result_plot_dir     = '{_fwd(os.path.join(save_dir, 'test', 'result_plots'))}'\n",
        '\n',
        '        # ── Dataset roots (split sub-dirs are handled in each dataset class) ─\n',
        f"        self.lasot_dir       = '{_fwd(os.path.join(data_dir, 'lasot'))}'\n",
        f"        self.tnl2k_dir       = '{_fwd(os.path.join(data_dir, 'tnl2k'))}'\n",
        f"        self.vasttrack_dir   = '{_fwd(os.path.join(data_dir, 'vasttrack'))}'\n",
        f"        self.otb99_lang_dir    = '{_fwd(os.path.join(data_dir, 'otb99_lang'))}'\n",
        f"        self.mgit_dir        = '{_fwd(os.path.join(data_dir, 'mgit'))}'\n",
    ]

    with open(config_path, 'w') as f:
        f.writelines(lines)

    print(f'Created {config_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate lib/local_config.py with dataset paths')
    parser.add_argument('--workspace_dir', type=str, required=True, help='Project root directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing all datasets')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory for saving results and checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_local_config(
        workspace_dir=os.path.realpath(args.workspace_dir),
        data_dir=os.path.realpath(args.data_dir),
        save_dir=os.path.realpath(args.save_dir),
    )
