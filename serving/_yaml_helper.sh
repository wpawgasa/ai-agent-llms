#!/usr/bin/env bash
# Shared YAML parsing helper sourced by all launcher scripts.
#
# Provides parse_yaml() — reads a dot-delimited key path from the YAML
# config file referenced by CONFIG_FILE (must be set by the sourcing script).
#
# Usage (after sourcing):
#   VALUE=$(parse_yaml "serving.gpu_memory_utilization" "0.90")

parse_yaml() {
    python3 - "$CONFIG_FILE" "$1" "$2" <<'EOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
keys = sys.argv[2].split('.')
default = sys.argv[3] if len(sys.argv) > 3 else ''
val = cfg
for k in keys:
    if isinstance(val, dict) and k in val:
        val = val[k]
    else:
        val = default
        break
print(val)
EOF
}
