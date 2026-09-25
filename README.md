# Wyoming Speech-to-Text on Coralboard SL2619 Torq NPU

A high-performance Speech-to-Text (STT) server implementing the **Wyoming protocol** for **Home Assistant**, accelerated on the **Synaptics Coralboard (Astra SL2619 SoC)** using its integrated **Torq NPU** (Google Coral Kelvin RISC-V ML core, 1 TOPS) running the **Moonshine** edge speech recognition model.

Built specifically for **Python 3.12.9** on **Linux aarch64**.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                              Home Assistant                             |
|               (Voice Satellite / Assist Pipeline / App)                 |
+------------------------------------+------------------------------------+
                                     |
                                     | Wyoming Protocol (TCP :10300)
                                     v
+-------------------------------------------------------------------------+
|                    Wyoming STT Server (Python 3.12.9)                   |
|                                                                         |
|  +-------------------------+         +-------------------------------+  |
|  |     Wyoming Server      |<------->|       WyomingSTTHandler       |  |
|  | (Describe / Transcribe) |         | (AudioStream / Chunk / Stop)  |  |
|  +-------------------------+         +---------------+---------------+  |
|                                                      |                  |
|                                         Raw PCM 16-bit 16kHz            |
|                                                      v                  |
|                                      +-------------------------------+  |
|                                      |      Audio Preprocessing      |  |
|                                      | (Float32 Resample/Normalize)  |  |
|                                      +---------------+---------------+  |
|                                                      |                  |
|                                                      v                  |
|                                      +-------------------------------+  |
|                                      |      TorqSTTEngine (NPU)      |  |
|                                      |  - torq_runtime               |  |
|                                      |  - VMFBInferenceRunner        |  |
|                                      |  - Coralboard Torq NPU        |  |
|                                      +-------------------------------+  |
+-------------------------------------------------------------------------+
```

### Key Capabilities
- **Native Wyoming Protocol**: Direct plug-and-play integration with Home Assistant's local voice control (Assist).
- **Coralboard SL2619 Acceleration**: Dedicated NPU inference using Synaptics' `torq_runtime` and pre-compiled Moonshine `.vmfb` artifacts.
- **Low Latency**: Dynamic audio scaling that processes only the spoken duration, yielding 150–300 ms response latency for typical smart home commands.

---

## Hardware & Environment Requirements

- **Device**: Synaptics Coralboard / Astra Machina SL2619 Dev Kit
- **SoC**: Synaptics Astra SL2619 (Dual Cortex-A55 @ 2.0 GHz, Torq 1 TOPS NPU)
- **OS**: Linux aarch64 (Yocto or Debian-based)
- **Python**: 3.12.9
- **Network**: Port `10300` accessible to your Home Assistant instance

---

## USB Networking Setup (Host & Coralboard)

Use [`tools/setup_usb_network.sh`](file:///Users/stephen/Projects/coral-sl2619-wisper/tools/setup_usb_network.sh) to bridge the Coralboard USB adapter to your host's LAN so the board receives an IP address directly from your network router via DHCP:

### 1. On Your Host Computer (macOS or Linux)
Connect the USB-C cable to the board's OTG port, then bridge the network interface:
```bash
sudo bash tools/setup_usb_network.sh --bridge
```
*Creates a network bridge (`bridge0` on macOS / `br0` on Linux) joining your host's network connection and the USB interface. The board is now on the same network as your host and router.*

### 2. On the Coralboard
Run on the Coralboard (via console or local shell):
```bash
sudo bash tools/setup_usb_network.sh --board
```
*Requests an IP address via DHCP directly from your local router on `usb0` and sets up persistent networking.*

---

## Remote Installation on Coralboard SL2619

### 1. Copy Project to Coralboard
From your development machine, transfer the project directory to the Coralboard:
```bash
scp -r coral-sl2619-stt root@<CORALBOARD_IP>:/home/root/
```

### 2. Run Automated Setup
SSH into the Coralboard and execute the automated setup script:
```bash
ssh root@<CORALBOARD_IP>
cd /home/root/coral-sl2619-stt
chmod +x install.sh
./install.sh
```

The script will:
1. Verify Python 3.12 runtime.
2. Initialize a virtual environment at `.venv` with access to system site packages.
3. Install dependencies from `requirements.txt`.
4. Install the official Synaptics Torq Runtime wheel for Python 3.12 (`cp312-manylinux_2_28_aarch64`).
5. Download the Moonshine tokenizer and pre-compiled Torq NPU `.vmfb` models into `models/`.

### 3. Downloading Models Manually (Optional)
You can also re-run model downloads anytime using:
```bash
python3 tools/download_models.py --models-dir models
```

---

## Running the Server

### Option A: Manual Launch (Testing & Debugging)

Activate the virtual environment and start the server:
```bash
source .venv/bin/activate

# Run on Torq NPU:
python3 -m coral_stt \
    --host 0.0.0.0 \
    --port 10300 \
    --model-path models/encode.vmfb \
    --decoder-path models/decode.vmfb \
    --vocab-path models/tokenizer.json \
    --debug
```

### Option B: Run as a Persistent Systemd Service

To keep the service running automatically in the background across reboots:

1. Copy the unit file:
```bash
sudo cp wyoming-coral-stt.service /etc/systemd/system/
```
2. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now wyoming-coral-stt
```
3. Check status and live logs:
```bash
sudo systemctl status wyoming-coral-stt
sudo journalctl -u wyoming-coral-stt -f
```

---

## Testing the Server

Verify the service directly on the Coralboard using `tools/test_client.py`:

```bash
# Test with synthetic test tone
python3 tools/test_client.py --host 127.0.0.1 --port 10300

# Test with a real WAV voice recording
python3 tools/test_client.py --host 127.0.0.1 --port 10300 --wav sample_voice.wav
```

---

## Adding to Home Assistant

Once the server is running on your Coralboard SL2619:

1. Open **Home Assistant** in your browser.
2. Go to **Settings** > **Devices & Services**.
3. Click **Add Integration** in the bottom right corner.
4. Search for and select **Wyoming Protocol**.
5. Configure the connection:
   - **Host**: IP address of your Coralboard (e.g. `192.168.1.150`)
   - **Port**: `10300`
6. Click **Submit**. Home Assistant will immediately discover the `coral-sl2619-stt` speech-to-text service.
7. Navigate to **Settings** > **Voice Assistants** > **Assist**:
   - Under **Speech-to-text**, select your new Coralboard NPU engine.
8. Test your voice satellite or use the Assist dialog to issue voice commands!
