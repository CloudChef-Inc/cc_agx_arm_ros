# ZED Body Streamer

Standalone Python script that runs on the ZedBox (Ubuntu 22.04). Captures skeleton data from a ZED Mini camera using the ZED SDK's body tracking and streams it as newline-delimited JSON over TCP.

No ROS 2 needed. No build step.

## Prerequisites

- ZED SDK installed (includes `pyzed` Python module)
- ZED Mini camera connected
- Python 3.8+
- `pip install numpy`

## Run

```bash
python3 zed_body_streamer.py --port 9090
```

## Test from another machine

```bash
nc <zedbox-ip> 9090
```

You should see JSON lines streaming at ~30 Hz when a person is visible.

## JSON format

```json
{"body_id":0,"confidence":0.92,"timestamp_ms":1776400000000,"keypoints":{"LEFT_SHOULDER":{"pos":[0.1,0.5,-1.2],"ori":[0,0,0,1],"conf":0.95},...}}
```

When no body is detected:
```json
{"body_id":-1,"confidence":0.0,"timestamp_ms":...,"keypoints":{}}
```
