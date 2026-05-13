# Path configuration is now centralized in lib/local_config.py.
# This file is kept for compatibility but is no longer read directly.
from lib.local_config import EnvSettings

def local_env_settings():
    return EnvSettings()
