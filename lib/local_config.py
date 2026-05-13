import os

class EnvSettings:
    def __init__(self):
        # ── Workspace & output ─────────────────────────────────────────────
        self.workspace_dir       = '/Tracker/MVLM'
        self.tensorboard_dir     = '/Tracker/MVLM/tensorboard'
        self.results_dir         = '/Tracker/MVLM/output/test/tracking_results'
        self.result_plot_dir     = '/Tracker/MVLM/output/test/result_plots'
        self.save_dir            = '/Tracker/MVLM/output'

        # ── Dataset roots (split sub-dirs are handled in each dataset class) ─
        self.lasot_dir       = '/Tracker/MVLM/data/lasot'
        self.tnl2k_dir       = '/Tracker/MVLM/data/tnl2k'
        self.vasttrack_dir   = '/Tracker/MVLM/data/vasttrack'
        self.otb99_lang_dir    = '/Tracker/MVLM/data/otb99_lang'
        self.mgit_dir        = '/Tracker/MVLM/data/mgit'