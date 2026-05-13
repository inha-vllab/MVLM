def env_settings():
    try:
        from lib.local_config import EnvSettings
        return EnvSettings()
    except ImportError:
        raise RuntimeError(
            'lib/local_config.py not found. '
            'Run: python tracking/create_default_local_file.py '
            '--workspace_dir <dir> --data_dir <dir> --save_dir <dir>'
        )
