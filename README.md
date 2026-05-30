# TurtleBot3 Autonomous Card Scanner

A ROS 2 node that drives a TurtleBot3 Burger along a row of playing cards,
memorises each one, waits while a person flips some of them, and then
revisits the cards to report which were flipped.

## Authors

| Name              | Student ID |
|-------------------|------------|
| Harshal Balar     | 70490214   |
| Prins Kathiriya   | 70490032   |
| Nehal Vaghasiya   | 70490292   |
| Latifa Sassi      | 70499382   |

## How It Works

1. The robot drives forward 25 cm, turns 90 degrees left to face a card,
   scans it, turns back, and repeats for all 5 cards (learn pass).
2. It turns away and waits 10 seconds while you flip some cards.
3. It turns back and walks the row in reverse, scanning each card again
   (test pass).
4. For each card, it compares the current scan against the memorised one
   using ORB feature matching with a RANSAC affine fit. If the rotation
   angle is close to 180 degrees, the card is reported as flipped.
5. After the last test scan, the robot drives back to its starting position.

Vision runs only while the robot is stationary. This avoids motion blur
and camera buffer lag entirely.

## Prerequisites

### Hardware

- TurtleBot3 Burger with Raspberry Pi 4
- Raspberry Pi Camera Module (connected via CSI ribbon cable)
- Laptop on the same WiFi network as the robot
- 5 standard playing cards laid in a straight line on the floor

### Software

Both the Pi and the laptop need:

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill

The laptop also needs:

- Python 3.10+
- OpenCV 4.5+ (`sudo apt install python3-opencv`)
- cv_bridge (`sudo apt install ros-humble-cv-bridge`)

## Setup

### 1. Create the workspace (laptop)

```bash
mkdir -p ~/cv_ws/src
cd ~/cv_ws/src
ros2 pkg create card_vision --build-type ament_python --dependencies rclpy sensor_msgs geometry_msgs nav_msgs cv_bridge
```

### 2. Add the source file

Copy `card_scanner_route.py` into the package:

```bash
cp card_scanner_route.py ~/cv_ws/src/card_vision/card_vision/card_scanner_route.py
```

### 3. Register the entry point

Open `~/cv_ws/src/card_vision/setup.py` and add the following inside the
`console_scripts` list in `entry_points`:

```python
'card_scanner_route = card_vision.card_scanner_route:main',
```

### 4. Build

```bash
cd ~/cv_ws
colcon build --packages-select card_vision
source install/setup.bash
```

### 5. Set the ROS domain ID

Both machines must use the same domain ID, and it must be different from
any other robot on the same network:

```bash
# On the Pi (over SSH)
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
source ~/.bashrc

# On the laptop
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
source ~/.bashrc
```

## Running

### Step 1: Start the robot (SSH into the Pi)

```bash
# Terminal 1 - robot bringup
ros2 launch turtlebot3_bringup robot.launch.py

# Terminal 2 - camera
ros2 run v4l2_camera v4l2_camera_node \
    --ros-args -p image_size:="[640,480]" -p brightness:=33
```

### Step 2: Run the scanner (on the laptop)

```bash
source ~/cv_ws/install/setup.bash
ros2 run card_vision card_scanner_route
```

Two OpenCV popup windows will appear: a live camera view and a memory grid
showing the learned cards. The terminal logs every action and prints the
test results at the end.

### Step 3: Watch the output

During the test pass, each card produces a log line like:

```
TEST RESULT: C5 R -> FLIPPED!  [angle=178 deg matches=87 matched=True]
TEST RESULT: C4 B -> untouched [angle=2 deg   matches=92 matched=True]
```

Cards detected as flipped are shown with a red border in the memory grid
popup.

### Step 4: Stop

Press `Ctrl+C` in the laptop terminal. The robot stops automatically.

## ROS 2 Topics

| Topic        | Type                    | Direction   | Purpose                        |
|--------------|-------------------------|-------------|--------------------------------|
| `/image_raw` | `sensor_msgs/Image`     | subscribed  | Camera frames from the Pi      |
| `/odom`      | `nav_msgs/Odometry`     | subscribed  | Wheel odometry from the Pi     |
| `/cmd_vel`   | `geometry_msgs/Twist`   | published   | Velocity commands to the robot |

## Project Structure

```
cv_ws/
  src/
    card_vision/
      card_vision/
        __init__.py
        card_scanner_route.py    # the main node (this file)
      setup.py
      package.xml
```

The entire project is a single Python file. There are no launch files,
config files, or external dependencies beyond ROS 2 and OpenCV.

## Configuration

All tunable parameters are constants at the top of `card_scanner_route.py`.
The most commonly adjusted ones:

| Constant         | Default | What it controls                              |
|------------------|---------|-----------------------------------------------|
| `TARGET_CARDS`   | 5       | Number of cards to learn and test             |
| `STEP_DIST`      | 0.25 m  | Distance between adjacent cards               |
| `TURN_ANGLE`     | 90 deg  | Turn angle to face each card                  |
| `DRIVE_SPEED`    | 0.06    | Forward speed in m/s                          |
| `BREAK_SECONDS`  | 10.0    | Seconds to wait while cards are flipped       |
| `SHARPNESS_MIN`  | 25.0    | Laplacian variance threshold for sharp frames |
| `POPUP_ENABLE`   | True    | Show the OpenCV popup windows                 |

## Troubleshooting

**Robot moves erratically or jumps to random positions**

Another robot on the same WiFi is using the same `ROS_DOMAIN_ID`. Change
yours to a unique number on both the Pi and the laptop, then restart.

**Camera frames are very laggy (10+ seconds old)**

This is normal on a slow WiFi link. The frame freshness gate in the code
automatically discards stale frames and waits for fresh ones. The robot
will just take longer to complete each scan.

**No card detected during a SCAN (robot holds position forever)**

Check the terminal output. It prints which detection gate rejected the
frame (area, aspect ratio, fill ratio, etc.). Common causes: the card is
too far from the camera, the lighting is too dim, or the card is outside
the bottom half of the frame.

**Odometry looks stale warning at startup**

The robot bringup was not restarted since the last run. The odometry
carries drift from the previous run. Restart the bringup on the Pi so
`/odom` resets to zero.

**Popup windows freeze or cause "not responding"**

Set `POPUP_ENABLE = False` at the top of the file. The popups are for
debugging only; the robot runs fine without them. All results are logged
to the terminal.

## References

- EdjeElectronics, "OpenCV Playing Card Detector", GitHub.
  https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector

- D. Dudas (MOGI-ROS), "Cognitive robotics with TurtleBot3, ROS 2 and
  OpenCV", GitHub.
  https://github.com/MOGI-ROS/Week-1-8-Cognitive-robotics
