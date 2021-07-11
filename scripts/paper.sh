#!/usr/bin/env bash
test -n "${DARWIN}" || exit
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

function paperchoice() {
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  python -c "import random; import glob; print(random.choice(glob.glob('${FILES}/Wall Papers/*')))"
}

if isuserdarwin.sh; then
  screens="$(wallpaper get --screen all | wc -l | xargs)"
  wallpaper set "$(paperchoice)" --scale auto --screen main
  ((screens -= 1))
  until [[ "${screens}" -eq 0 ]]; do
    wallpaper set "$(paperchoice)" --scale auto --screen "${screens}"
    ((screens -= 1))
  done
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime starting
