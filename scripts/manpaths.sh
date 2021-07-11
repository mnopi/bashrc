#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

sudo mkdir -p "/etc/manpaths.d"
file="/etc/manpaths.d/$( basename "${BASH_SOURCE[0]}" )"
if sudo tee "${file}" >/dev/null <<EOT; then
${BREW}/../share/man
${MACDEV}/man
${ICLOUD}/man
EOT
  info.sh manpaths "${file}"
else
  error.sh manpaths "${file}"; exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting file
