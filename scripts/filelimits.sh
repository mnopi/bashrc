#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting


file=/Library/LaunchDaemons/limit.maxfiles.plist
if sudo tee "${file}" >/dev/null <<EOT; then
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>limit.maxfiles</string>
    <key>ProgramArguments</key>
    <array>
      <string>launchctl</string>
      <string>limit</string>
      <string>maxfiles</string>
      <string>65536</string>
      <string>524288</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ServiceIPC</key>
    <false/>
  </dict>
</plist>
EOT
  sudo chown root:wheel "${file}"
  sudo launchctl load -w "${file}" > /dev/null 2>&1
  info.sh maxfiles "$(sudo launchctl limit maxfiles)"
else
  error.sh maxfiles "${2}"; return 1
fi

unset starting file
