#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"


file=/Library/LaunchDaemons/limit.maxfiles.plist
if sudo tee "${file}" >/dev/null <<EOT
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
        "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>limit.maxfiles</string>
    <key>ProgramArguments</key>
    <array>
      <string>launchctl</string>
      <string>limit</string>
      <string>maxfiles</string>
      <string>${MAXFILES}</string>
      <string>${MAXFILES}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ServiceIPC</key>
    <false/>
  </dict>
</plist>
EOT
then
  sudo chown root:wheel "${file}"
  sudo launchctl load -w "${file}" > /dev/null 2>&1
  info.sh maxfiles "$(sudo launchctl limit maxfiles)"
else
  error.sh maxfiles "${2}"; return 1
fi

file=/Library/LaunchDaemons/limit.maxproc.plist
if sudo tee "${file}" >/dev/null <<EOT
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
        "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>limit.maxproc</string>
    <key>ProgramArguments</key>
    <array>
      <string>launchctl</string>
      <string>limit</string>
      <string>maxproc</string>
      <string>${MAXPROC}</string>
      <string>${MAXPROC}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ServiceIPC</key>
    <false/>
  </dict>
</plist>
EOT
then
  sudo chown root:wheel "${file}"
  sudo launchctl load -w "${file}" > /dev/null 2>&1
  info.sh ulimit "$(sudo launchctl limit maxproc)"
else
  error.sh ulimit "${2}"; return 1
fi


[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting file
