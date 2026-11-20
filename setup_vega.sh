#!/usr/bin/env bash
# Bring the Polaris Vega up and check it: network address, reachability, API.
# Safe to run any time (idempotent). Usage: ./setup_vega.sh
set -u

VEGA_IP=192.168.3.3
HOST_ADDR=192.168.3.10/24
IFACE=enp98s0
PROFILE=vega           # NetworkManager profile on $IFACE (the UR is on its own USB adapter, profile ur16e)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ok()   { printf '  \e[32mOK\e[0m    %s\n' "$*"; }
fix()  { printf '  \e[33mFIXED\e[0m %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m  %s\n' "$*"; exit 1; }

echo "Polaris Vega setup"

# 1. cable / link
[ "$(cat /sys/class/net/$IFACE/carrier 2>/dev/null)" = 1 ] \
    || fail "no link on $IFACE: check the cable and that the Vega's power supply is on"
ok "link up on $IFACE"

# 2. saved profile has the Vega subnet address (persists across reboots)
if nmcli -g ipv4.addresses con show "$PROFILE" | grep -q "${HOST_ADDR%/*}"; then
    ok "profile '$PROFILE' has $HOST_ADDR"
else
    nmcli con modify "$PROFILE" +ipv4.addresses "$HOST_ADDR" || fail "could not modify profile '$PROFILE'"
    fix "added $HOST_ADDR to profile '$PROFILE'"
fi

# 3. address is live on the interface right now
if ip -br addr show "$IFACE" | grep -q "${HOST_ADDR%/*}"; then
    ok "$IFACE has ${HOST_ADDR%/*}"
else
    nmcli dev modify "$IFACE" +ipv4.addresses "$HOST_ADDR" >/dev/null || fail "could not add address to $IFACE"
    fix "added ${HOST_ADDR%/*} to $IFACE"
fi

# 4. Vega reachable (it takes ~1 min to boot after power-on)
for i in $(seq 1 30); do
    ping -c1 -W1 "$VEGA_IP" >/dev/null && break
    [ "$i" = 1 ] && echo "        waiting for Vega at $VEGA_IP (up to 60 s)..."
    sleep 1
done
ping -c1 -W1 "$VEGA_IP" >/dev/null || fail "Vega not answering at $VEGA_IP: still booting, or IP changed?"
ok "Vega answers ping at $VEGA_IP"

# 5. API responds, no alerts
cd "$HERE" && python3 - "$VEGA_IP" <<'EOF' || exit 1
import sys
from ndi_tracking.vega import Vega
try:
    v = Vega(sys.argv[1]); v.init()
    ver = v.info().splitlines()
    alerts = v.command('GET Info.Status.Alerts').split('=')[1]
    v.sock.close()
except Exception as e:
    print(f'  \033[31mFAIL\033[0m  API on port 8765: {e}'); sys.exit(1)
print(f'  \033[32mOK\033[0m    {ver[0]}, {ver[1]}, {ver[3]}')
if alerts.strip() not in ('0', '0x00000000'):
    print(f'  \033[33mWARN\033[0m  tracker alerts = {alerts} (check NDI status LEDs)')
EOF

echo "Ready. Next: python3 -m ndi_tracking check   (or record --plot)"
