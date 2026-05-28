# Source: deploy/pipelines/config_loader.py — channel list parsing in load_config()
# Changes applied:
#   - Extracted _parse_channel_list() helper at module level (not a method — no self needed)
#   - Call site reduced from 6 lines to 2, and a future third field is one more line
#   - The parsing logic now lives in one place — change once, applies everywhere

def _parse_channel_list(value) -> List[int]:
    """Normalise a channel list from config.yaml — accepts a Python list or a comma-separated string."""
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return list(value)


# In load_config():
entry_exit_channels = _parse_channel_list(cfg.get("entry_exit_channels", []))
external_channels   = _parse_channel_list(cfg.get("external_channels",   []))
