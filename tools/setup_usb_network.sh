#!/usr/bin/env bash
# ==============================================================================
# Safe USB-Ethernet Networking Setup for Synaptics Coralboard SL2619 (Linux)
# Configures a dedicated, isolated point-to-point link with optional Internet Sharing (NAT).
# NEVER touches or modifies your host's primary network adapter or default gateway.
# ==============================================================================

set -euo pipefail

DEFAULT_HOST_IP="192.168.100.1"
DEFAULT_BOARD_IP="192.168.100.2"
DEFAULT_NETMASK="255.255.255.0"
DEFAULT_PREFIX="24"
DEFAULT_SUBNET="192.168.100.0/24"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $*"
}

log_ok() {
    echo -e "\033[1;32m[OK]\033[0m $*"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $*"
}

log_err() {
    echo -e "\033[1;31m[ERROR]\033[0m $*"
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        log_err "This command must be run with root privileges: sudo $0 $*"
        exit 1
    fi
}

detect_environment() {
    if [[ -f /etc/os-release ]] && grep -qiE "astra|synaptics|yocto|coral" /etc/os-release 2>/dev/null; then
        echo "board"
    else
        echo "host"
    fi
}

find_linux_usb_interface() {
    for candidate in $(ip -o link show | awk -F': ' '{print $2}'); do
        if [[ "$candidate" =~ ^(usb[0-9]+|enx[0-9a-f]+) ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# ------------------------------------------------------------------------------
# Emergency Network Restore (Host)
# ------------------------------------------------------------------------------
restore_host_network() {
    require_root
    log_info "Restoring host network configuration..."

    # Release any interfaces enslaved to br0
    if ip link show br0 >/dev/null 2>&1; then
        log_info "Removing interfaces from br0..."
        for member in $(ip -o link show master br0 2>/dev/null | awk -F': ' '{print $2}'); do
            log_info "  Detaching $member from br0..."
            ip link set "$member" nomaster 2>/dev/null || true
            ip link set "$member" up 2>/dev/null || true
        done
        ip link set br0 down 2>/dev/null || true
        ip link delete br0 type bridge 2>/dev/null || true
        log_ok "Removed bridge br0."
    fi

    # Restart host network services if available
    log_info "Restarting host network manager..."
    if systemctl is-active --quiet NetworkManager 2>/dev/null; then
        systemctl restart NetworkManager
        log_ok "NetworkManager restarted."
    elif systemctl is-active --quiet systemd-networkd 2>/dev/null; then
        systemctl restart systemd-networkd
        log_ok "systemd-networkd restarted."
    fi

    log_ok "Host network restoration complete."
}

# ------------------------------------------------------------------------------
# Board Side Configuration
# ------------------------------------------------------------------------------
configure_board() {
    require_root
    local iface="${1:-usb0}"

    log_info "Configuring Coralboard USB interface ($iface)..."

    ip link set "$iface" up 2>/dev/null || ifconfig "$iface" up 2>/dev/null || {
        log_err "Interface $iface not found! Verify USB-C OTG cable is connected."
        exit 1
    }

    log_info "Assigning static IP $DEFAULT_BOARD_IP/$DEFAULT_PREFIX to $iface..."
    ip addr flush dev "$iface" 2>/dev/null || true
    ip addr add "${DEFAULT_BOARD_IP}/${DEFAULT_PREFIX}" dev "$iface" 2>/dev/null || ifconfig "$iface" "$DEFAULT_BOARD_IP" netmask "$DEFAULT_NETMASK"
    
    # Route all outbound traffic to the Host gateway
    log_info "Setting default gateway to $DEFAULT_HOST_IP..."
    ip route replace default via "$DEFAULT_HOST_IP" dev "$iface" 2>/dev/null || route add default gw "$DEFAULT_HOST_IP" "$iface" 2>/dev/null || true

    # Configure DNS
    log_info "Configuring DNS servers (1.1.1.1, 8.8.8.8)..."
    echo -e "nameserver 1.1.1.1\nnameserver 8.8.8.8" > /etc/resolv.conf

    log_ok "Coralboard $iface configured with IP: $DEFAULT_BOARD_IP (Gateway: $DEFAULT_HOST_IP)"
    
    # Test ping to host
    log_info "Testing connectivity to host ($DEFAULT_HOST_IP)..."
    if ping -c 2 -W 2 "$DEFAULT_HOST_IP" 2>/dev/null; then
        log_ok "Host is reachable!"
    else
        log_warn "Host not responding. Ensure host side is configured: 'sudo $0 --host'"
    fi

    # Test internet access
    log_info "Testing internet connectivity..."
    if ping -c 2 -W 2 1.1.1.1 2>/dev/null; then
        log_ok "Internet connectivity verified on the Coralboard!"
    else
        log_warn "No internet access yet. Enable NAT on your host: 'sudo $0 --host --nat'"
    fi
}

# ------------------------------------------------------------------------------
# Host Side Configuration (Isolated Point-to-Point + Optional NAT)
# ------------------------------------------------------------------------------
configure_host() {
    require_root
    local usb_iface="${1:-}"
    local enable_nat="${2:-false}"

    if [[ -z "$usb_iface" ]]; then
        usb_iface=$(find_linux_usb_interface || true)
    fi

    if [[ -z "$usb_iface" ]]; then
        log_err "No Coralboard USB Ethernet interface detected (usbX / enxX)."
        log_info "Verify the USB-C cable is connected to the Coralboard's OTG port."
        log_info "Or specify the interface explicitly: sudo $0 --host <interface_name>"
        exit 1
    fi

    log_info "Configuring host USB interface $usb_iface with IP $DEFAULT_HOST_IP..."
    ip link set "$usb_iface" up
    ip addr flush dev "$usb_iface" 2>/dev/null || true
    ip addr add "${DEFAULT_HOST_IP}/${DEFAULT_PREFIX}" dev "$usb_iface"

    log_ok "Host USB interface $usb_iface configured with IP $DEFAULT_HOST_IP"
    log_ok "Your host's primary LAN/Wi-Fi connection remains completely untouched."

    # Optional: Enable NAT Internet Sharing & Forwarding Rules
    if [[ "$enable_nat" == "true" ]]; then
        local uplink
        uplink=$(ip route show default 2>/dev/null | awk '{print $5}' | head -n1 || true)

        log_info "Enabling IPv4 forwarding..."
        sysctl -w net.ipv4.ip_forward=1 >/dev/null 2>&1

        if command -v iptables >/dev/null 2>&1; then
            log_info "Applying iptables rules to allow traffic from $usb_iface to the internet..."

            # 1. Allow input from USB interface to host services (DNS, ICMP ping)
            iptables -C INPUT -i "$usb_iface" -j ACCEPT 2>/dev/null || \
            iptables -I INPUT 1 -i "$usb_iface" -j ACCEPT 2>/dev/null || true

            # 2. Forwarding: Allow traffic originating from USB interface out to the internet
            iptables -C FORWARD -i "$usb_iface" ! -o "$usb_iface" -j ACCEPT 2>/dev/null || \
            iptables -I FORWARD 1 -i "$usb_iface" ! -o "$usb_iface" -j ACCEPT 2>/dev/null || true

            if [[ -n "$uplink" ]]; then
                iptables -C FORWARD -i "$usb_iface" -o "$uplink" -j ACCEPT 2>/dev/null || \
                iptables -I FORWARD 1 -i "$usb_iface" -o "$uplink" -j ACCEPT 2>/dev/null || true
            fi

            # 3. Forwarding: Allow established/related return traffic back to the Coralboard
            iptables -C FORWARD -o "$usb_iface" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || \
            iptables -I FORWARD 1 -o "$usb_iface" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || \
            iptables -I FORWARD 1 -o "$usb_iface" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true

            # 4. NAT Masquerade: Rewrite source IP for outbound packets from the Coralboard subnet
            iptables -t nat -C POSTROUTING -s "$DEFAULT_SUBNET" ! -o "$usb_iface" -j MASQUERADE 2>/dev/null || \
            iptables -t nat -I POSTROUTING 1 -s "$DEFAULT_SUBNET" ! -o "$usb_iface" -j MASQUERADE 2>/dev/null || true

            if [[ -n "$uplink" ]]; then
                iptables -t nat -C POSTROUTING -s "$DEFAULT_SUBNET" -o "$uplink" -j MASQUERADE 2>/dev/null || \
                iptables -t nat -I POSTROUTING 1 -s "$DEFAULT_SUBNET" -o "$uplink" -j MASQUERADE 2>/dev/null || true
            fi

            # 5. Firewalld integration (if active on Fedora/RHEL/CentOS)
            if command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
                log_info "firewalld is active; trusting $usb_iface and enabling masquerade..."
                firewall-cmd --zone=trusted --add-interface="$usb_iface" 2>/dev/null || true
                firewall-cmd --add-masquerade 2>/dev/null || true
            fi

            log_ok "iptables rules applied: Internet traffic from $usb_iface is now allowed and masqueraded."
        else
            log_warn "iptables not found; please install iptables to enable automatic NAT."
        fi
    fi

    log_info "Pinging Coralboard at $DEFAULT_BOARD_IP..."
    if ping -c 2 -W 2 "$DEFAULT_BOARD_IP" 2>/dev/null; then
        log_ok "Coralboard ($DEFAULT_BOARD_IP) is responding!"
        log_ok "Connect via SSH: ssh root@$DEFAULT_BOARD_IP"
    else
        log_warn "Coralboard ($DEFAULT_BOARD_IP) not responding yet."
        log_info "Run on the Coralboard: sudo bash tools/setup_usb_network.sh --board"
    fi
}

# ------------------------------------------------------------------------------
# Entrypoint & CLI Parsing
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
Usage: sudo $0 [MODE] [OPTIONS]

Safely configures USB Ethernet networking between a Linux host and Coralboard SL2619.

Modes:
  --host [USB_IFACE] [--nat]
      Configures host USB port with IP $DEFAULT_HOST_IP.
      Add --nat to share host internet connection without touching host network.

  --board [IFACE]
      Configures Coralboard with IP $DEFAULT_BOARD_IP, sets host as default gateway,
      and configures DNS.

  --restore
      Emergency command to tear down any leftover bridge interfaces on host.

Examples:
  # 1. Share internet from host to Coralboard:
  sudo $0 --host --nat

  # 2. On the Coralboard:
  sudo $0 --board

  # 3. Test internet on the Coralboard:
  curl -I https://huggingface.co
EOF
}

TARGET_MODE=""
TARGET_IFACE=""
ENABLE_NAT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            TARGET_MODE="host"
            shift
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                TARGET_IFACE="$1"; shift
            fi
            ;;
        --board)
            TARGET_MODE="board"
            shift
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                TARGET_IFACE="$1"; shift
            fi
            ;;
        --nat)
            ENABLE_NAT=true
            shift
            ;;
        --restore)
            TARGET_MODE="restore"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_err "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_MODE" ]]; then
    ENV_TYPE=$(detect_environment)
    if [[ "$ENV_TYPE" == "board" ]]; then
        TARGET_MODE="board"
    else
        TARGET_MODE="host"
    fi
fi

case "$TARGET_MODE" in
    host)
        configure_host "$TARGET_IFACE" "$ENABLE_NAT"
        ;;
    board)
        configure_board "${TARGET_IFACE:-usb0}"
        ;;
    restore)
        restore_host_network
        ;;
esac
