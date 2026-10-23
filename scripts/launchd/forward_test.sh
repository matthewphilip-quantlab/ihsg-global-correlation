#!/bin/zsh
# Pasang / copot jadwal launchd untuk forward test (scripts/daily_signal.py).
#   scripts/launchd/forward_test.sh install    # Senin-Jumat 09:15 waktu lokal Mac (Australia/Melbourne)
#   scripts/launchd/forward_test.sh uninstall
#   scripts/launchd/forward_test.sh status
#   scripts/launchd/forward_test.sh run-now    # jalankan sekali sekarang lewat launchd (untuk uji)
# Output & error: <repo>/logs/daily_signal.out.log dan .err.log (tidak di-commit).
set -euo pipefail

LABEL="com.ihsg-global-correlation.daily-signal"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${IHSG_PYTHON:-$HOME/anaconda3/envs/ihsg-corr/bin/python}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
HOUR=9
MINUTE=15

write_plist() {
  mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"
  {
    cat <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$REPO/scripts/daily_signal.py</string></array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>StandardOutPath</key><string>$REPO/logs/daily_signal.out.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/daily_signal.err.log</string>
  <key>RunAtLoad</key><false/>
  <key>StartCalendarInterval</key>
  <array>
XML
    for wd in 1 2 3 4 5; do
      echo "    <dict><key>Weekday</key><integer>$wd</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>$MINUTE</integer></dict>"
    done
    cat <<XML
  </array>
</dict>
</plist>
XML
  } > "$PLIST"
  plutil -lint "$PLIST" >/dev/null
}

case "${1:-}" in
  install)
    [[ -x "$PY" ]] || { echo "Python tidak ditemukan: $PY (set IHSG_PYTHON)"; exit 1; }
    "$PY" "$REPO/scripts/daily_signal.py" --check-schedule
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    write_plist
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Terpasang: $PLIST (Senin-Jumat $HOUR:$(printf %02d $MINUTE) waktu lokal Mac)"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Dicopot: $LABEL"
    ;;
  status)
    launchctl print "$DOMAIN/$LABEL" | grep -E "state|last exit code|runs|path =" || echo "Tidak terpasang."
    ;;
  run-now)
    launchctl kickstart -k "$DOMAIN/$LABEL"
    echo "Dijalankan; lihat $REPO/logs/daily_signal.out.log"
    ;;
  *)
    echo "Pemakaian: $0 install|uninstall|status|run-now"; exit 2 ;;
esac
