# Source: deploy/pipelines/config_loader.py — channel list parsing in load_config()
# Anti-pattern illustrated:
#   - Verbatim copy-paste of a 3-line normalisation block.
#     If a third channel-list field is added (e.g. restricted_channels), it gets copied again.
#     If the parsing logic changes (e.g. support semicolons), it must be changed in N places.

entry_exit_channels = cfg.get("entry_exit_channels", [])
if isinstance(entry_exit_channels, str):
    entry_exit_channels = [int(x.strip()) for x in entry_exit_channels.split(",") if x.strip()]

external_channels = cfg.get("external_channels", [])
if isinstance(external_channels, str):
    external_channels = [int(x.strip()) for x in external_channels.split(",") if x.strip()]
