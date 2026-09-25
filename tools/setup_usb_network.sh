#!/usr/bin/env bash
# ==============================================================================
# USB-Ethernet Networking Setup for Synaptics Coralboard SL2619
# Exposes the Coralboard to the same network as the host via bridging,
# allowing the board to obtain an IP address directly from the LAN's DHCP router.
# ==============================================================================

set -euo pipefail

DEFAULT_HOST_IP="192.168.100.1"
DEFAULT_BOARD_IP="192.168.100.2"
DEFAULT_NETMASK="255.255.255.0"
DEFAULT_PREFIX="24"

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
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "host-mac"
    else
        echo "host-linux"
    fi
}

find_mac_usb_interface() {
    local target=""
    while read -r line; do
        if echo "$line" | grep -qiE "USB|CDC|NCM|Ethernet|Gadget"; then
            read -r dev_line
            target=$(echo "$dev_line" | awk '{print $NF}')
            if [[ -n "$target" ]]; then
                echo "$target"
                return 0
            fi
        fi
    done < <(networksetup -listallhardwareports 2>/dev/null || true)
    return 1
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
# Board Side Configuration
# ------------------------------------------------------------------------------
configure_board() {
    require_root
    local iface="${1:-usb0}"

    log_info "Bringing up $iface on the Coralboard..."

    ip link set "$iface" up 2>/dev/null || ifconfig "$iface" up 2>/dev/null || {
        log_err "Interface $iface not found! Verify USB-C OTG cable is connected."
        exit 1
    }

    log_info "Requesting DHCP lease on $iface from the network..."
    local dhcp_success=false

    if command -v udhcpc >/dev/null 2>&1; then
        if udhcpc -i "$iface" -n -q -t 5 -T 2; then
            dhcp_success=true
        fi
    elif command -v dhclient >/dev/null 2>&1; then
        if dhclient -1 -v "$iface"; then
            dhcp_success=true
        fi
    fi

    if [[ "$dhcp_success" == "true" ]]; then
        local assigned_ip
        assigned_ip=$(ip -4 addr show dev "$iface" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1 || ifconfig "$iface" | grep 'inet ' | awk '{print $2}' || true)
        log_ok "Coralboard acquired DHCP address: $assigned_ip"
    else
        log_warn "No DHCP response received from the network on $iface."
        log_info "Ensure the host has bridged the USB adapter to the LAN: 'sudo $0 --bridge'"
        log_info "Falling back to static IP: $DEFAULT_BOARD_IP..."

        ip addr flush dev "$iface" 2>/dev/null || true
        ip addr add "${DEFAULT_BOARD_IP}/${DEFAULT_PREFIX}" dev "$iface" 2>/dev/null || ifconfig "$iface" "$DEFAULT_BOARD_IP" netmask "$DEFAULT_NETMASK"
        ip route replace default via "$DEFAULT_HOST_IP" dev "$iface" 2>/dev/null || route add default gw "$DEFAULT_HOST_IP" "$iface" 2>/dev/null || true

        if [[ ! -s /etc/resolv.conf ]] || ! grep -q "nameserver" /etc/resolv.conf; then
            echo -e "nameserver 1.1.1.1\nnameserver 8.8.8.8" > /etc/resolv.conf
        fi
        log_ok "Configured with static IP: $DEFAULT_BOARD_IP"
    fi

    # Persistent systemd-networkd configuration
    if [[ -d /etc/systemd/network ]]; then
        log_info "Saving persistent systemd-network configuration to /etc/systemd/network/10-${iface}.network..."
        cat <<EOF > "/etc/systemd/network/10-${iface}.network"
[Match]
Name=${iface}

[Network]
DHCP=yes

[DHCPv4]
UseDNS=yes
UseRoutes=yes
EOF
        systemctl restart systemd-networkd 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------
# Host Side: Bridge USB to LAN (Exposing Board to Host Network)
# ------------------------------------------------------------------------------
configure_bridge_mac() {
    require_root
    local uplink="${1:-en0}"
    local usb_iface="${2:-}"

    if [[ -z "$usb_iface" ]]; then
        usb_iface=$(find_mac_usb_interface || true)
    fi

    if [[ -z "$usb_iface" ]]; then
        log_err "Could not detect USB Ethernet interface. Usage: sudo $0 --bridge <uplink> <usb_iface>"
        exit 1
    fi

    log_info "Bridging $uplink and $usb_iface on macOS..."

    # Reset bridge0 if already present
    ifconfig bridge0 destroy 2>/dev/null || true

    # Create Layer-2 bridge
    ifconfig bridge0 create
    ifconfig bridge0 addm "$uplink" addm "$usb_iface"
    ifconfig "$usb_iface" up
    ifconfig bridge0 up

    log_ok "Bridge 'bridge0' created joining $uplink and $usb_iface."
    log_ok "The Coralboard is now exposed directly to your host's local network."
    log_info "Run on the Coralboard to request an IP from your LAN DHCP router:"
    log_info "  sudo bash tools/setup_usb_network.sh --board"
}

configure_bridge_linux() {
    require_root
    local uplink="${1:-}"
    local usb_iface="${2:-}"

    if [[ -z "$uplink" ]]; then
        uplink=$(ip route show default 2>/dev/null | awk '{print $5}' | head -n1 || true)
    fi

    if [[ -z "$usb_iface" ]]; then
        usb_iface=$(find_linux_usb_interface || true)
    fi

    if [[ -z "$uplink" || -z "$usb_iface" ]]; then
        log_err "Could not identify network interfaces. Usage: sudo $0 --bridge <uplink> <usb_iface>"
        exit 1
    fi

    log_info "Bridging $uplink and $usb_iface into 'br0' on Linux..."

    ip link add name br0 type bridge 2>/dev/null || true
    ip link set "$usb_iface" master br0
    ip link set "$uplink" master br0
    ip link set "$usb_iface" up
    ip link set br0 up

    log_ok "Linux bridge 'br0' created joining $uplink and $usb_iface."
    log_ok "The Coralboard is now exposed directly to your host's local network."
    log_info "Run on the Coralboard to request an IP from your LAN DHCP router:"
    log_info "  sudo bash tools/setup_usb_network.sh --board"
}

# ------------------------------------------------------------------------------
# Host Side: Point-to-Point Static IP (No Bridge)
# ------------------------------------------------------------------------------
configure_host_static() {
    require_root
    local usb_iface="${1:-}"

    if [[ "$(uname -s)" == "Darwin" ]]; then
        if [[ -z "$usb_iface" ]]; then
            usb_iface=$(find_mac_usb_interface || true)
        fi
        if [[ -z "$usb_iface" ]]; then
            log_err "No USB interface detected. Usage: sudo $0 --host <interface>"
            exit 1
        fi
        log_info "Assigning static IP $DEFAULT_HOST_IP to $usb_iface..."
        ifconfig "$usb_iface" "$DEFAULT_HOST_IP" netmask "$DEFAULT_NETMASK" up
    else
        if [[ -z "$usb_iface" ]]; then
            usb_iface=$(find_linux_usb_interface || true)
        fi
        if [[ -z "$usb_iface" ]]; then
            log_err "No USB interface detected. Usage: sudo $0 --host <interface>"
            exit 1
        fi
        log_info "Assigning static IP $DEFAULT_HOST_IP to $usb_iface..."
        ip link set "$usb_iface" up
        ip addr flush dev "$usb_iface" 2>/dev/null || true
        ip addr add "${DEFAULT_HOST_IP}/${DEFAULT_PREFIX}" dev "$usb_iface"
    fi

    log_ok "Host interface $usb_iface configured with IP $DEFAULT_HOST_IP"
}

# ------------------------------------------------------------------------------
# Entrypoint & CLI Parsing
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
Usage: sudo $0 [MODE] [OPTIONS]

Configures USB Ethernet networking between a host machine (macOS/Linux)
and the Synaptics Coralboard SL2619.

Modes:
  --bridge [UPLINK] [USB_IFACE]
      Bridges the host's LAN network connection with the Coralboard USB adapter,
      exposing the Coralboard directly onto your local network so it acquires
      its IP address from your router's DHCP server.

  --host [USB_IFACE]
      Configures the host USB port with a static IP ($DEFAULT_HOST_IP)
      for direct point-to-point connections.

  --board [IFACE]
      Run directly on the Coralboard to acquire a DHCP lease from the LAN.

Examples:
  # 1. On your host machine (bridge USB adapter to LAN):
  sudo $0 --bridge

  # 2. On the Coralboard (acquire DHCP lease from router):
  sudo $0 --board
EOF
}

TARGET_MODE=""
PARAM1=""
PARAM2=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bridge)
            TARGET_MODE="bridge"
            shift
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                PARAM1="$1"; shift
            fi
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                PARAM2="$1"; shift
            fi
            ;;
        --host)
            TARGET_MODE="host"
            shift
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                PARAM1="$1"; shift
            fi
            ;;
        --board)
            TARGET_MODE="board"
            shift
            if [[ -n "${1:-}" && ! "$1" =~ ^-- ]]; then
                PARAM1="$1"; shift
            fi
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
        TARGET_MODE="bridge"
    fi
fi

case "$TARGET_MODE" in
    bridge)
        if [[ "$(uname -s)" == "Darwin" ]]; then
            configure_bridge_mac "$PARAM1" "$PARAM2"
        else
            configure_bridge_linux "$PARAM1" "$PARAM2"
        fi
        ;;
    host)
        configure_host_static "$PARAM1"
        ;;
    board)
        configure_board "${PARAM1:-usb0}"
        ;;
esac
