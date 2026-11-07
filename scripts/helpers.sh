#!/usr/bin/env bash

# Shared helpers for the install scripts. Source only.

# backup_if_changed DEST NEW BACKUP_DIR
# Warn and keep DEST as BACKUP_DIR/<name>.bak when installing NEW over it would change it.
backup_if_changed() {
    local dest="$1" new="$2" backup_dir="$3"
    if [[ -f "$dest" ]] && ! cmp -s "$dest" "$new"; then
        mkdir -p "$backup_dir"
        cp "$dest" "$backup_dir/$(basename "$dest").bak"
        echo "Warning: overwriting $dest; previous copy saved to $backup_dir/$(basename "$dest").bak" >&2
    fi
}
