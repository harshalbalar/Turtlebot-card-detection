"""
TurtleBot3 Autonomous Card Scanner
==================================

Course   : [Course Name / Code]
Project  : Card Memorisation and Flip Detection
Date     : 2026

Authors  : Harshal Balar - 70490214
           Prins Kathiriya - 70490032
           Nehal Vaghasiya - 70490292
           Latifa Sassi - 70499382

The robot drives along a row of playing cards, memorises each one, waits
for a person to flip some of them, and then revisits the cards to report
which were flipped. Vision runs only while the robot is stationary, which
avoids problems with motion blur and camera latency.

Flip detection uses ORB feature matching with a RANSAC affine fit; the
rotation angle of the fit tells us whether the card was rotated 180
degrees between the two scans.

Topics
------
    /image_raw   sensor_msgs/Image     subscribed (camera)
    /odom        nav_msgs/Odometry     subscribed (wheel odometry)
    /cmd_vel     geometry_msgs/Twist   published  (motion commands)

Usage
-----
On the robot (over SSH):
    ros2 run v4l2_camera v4l2_camera_node \\
        --ros-args -p image_size:="[640,480]" -p brightness:=33

On the laptop:
    ros2 run card_vision card_scanner_route

ROS_DOMAIN_ID must match on both machines, and should differ from any
other ROS 2 system sharing the network.
"""

import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TARGET_CARDS = 5

# Distance between cards, and the turn angle to face each card.
# In ROS, +angular.z is a left turn (counter-clockwise).
STEP_DIST  = 0.25      # m
TURN_ANGLE = 90.0      # deg

# Motion speeds for a TurtleBot3 Burger on smooth indoor flooring.
DRIVE_SPEED = 0.06     # m/s
TURN_SPEED  = 0.5      # rad/s (maximum)

# Turning at full speed up to the target overshoots due to wheel inertia.
# Inside the slow zone the speed scales down linearly so the robot creeps
# the last few degrees and lands on target.
TURN_SLOW_ZONE = math.radians(25)
TURN_MIN_SPEED = 0.07  # rad/s

DIST_TOL  = 0.015            # m
ANGLE_TOL = math.radians(1)  # rad

# Scan timing.
BREAK_SECONDS    = 10.0
SCAN_SETTLE_SECS = 2.0

# After the robot stops, the camera buffer still holds blurred frames
# from while it was moving. Capture only fires after this many consecutive
# sharp frames with a card detected.
SCAN_GOOD_FRAMES    = 2
SCAN_MISS_TOLERANCE = 3

# Sharpness threshold (variance of the Laplacian). Below this a frame is
# treated as too blurry to capture.
SHARPNESS_MIN = 25.0

# Card shape gates. The bands are wide enough to accept tilted and
# partially clipped cards.
CARD_ASPECT_MIN = 0.80
CARD_ASPECT_MAX = 3.00
CARD_MIN_FILL   = 0.45
CARD_MIN_AREA   = 150
CARD_MAX_AREA   = 120000

# Top fraction of the cropped (bottom-half) frame to ignore. This blanks
# out the wall / skirting that can appear above the floor.
DETECT_TOP_IGNORE = 0.18

# Optional interior-detail check. Off by default because it tends to
# reject red-suit cards on dim cameras (red ink barely shows up in
# luminance grayscale). Turn back on if blank floor patches start
# triggering false detections.
ENABLE_DETAIL_GATE = False
CARD_MIN_DETAIL    = 0.006

# Image enhancement. Bilateral denoising is done first so CLAHE and
# unsharp masking enhance the card and not the grain.
CLAHE_CLIP, CLAHE_TILE = 1.8, 8
GAMMA          = 1.10
UNSHARP_SIGMA  = 1.2
UNSHARP_AMOUNT = 0.4
DENOISE_ENABLE   = True
DENOISE_DIAMETER = 5
DENOISE_SIGMA    = 50

# The enhance + detect pipeline is expensive, so it only runs on every
# Nth image callback during a scan. Detection still happens several times
# a second, which is plenty for a stationary card.
SCAN_PROCESS_EVERY = 5
STREAM_W, STREAM_H = 480, 360

# Frame freshness gate. Each image carries a capture timestamp. Frames
# captured before the robot's last motion ended show the robot mid-turn
# and are discarded.
FRAME_FRESH_GUARD = 0.4   # s

# Display popups. The live popup is updated every Nth callback so the
# Qt event loop has time to draw between heavy frames.
POPUP_ENABLE    = True
POPUP_EVERY_NTH = 3

DEBUG_LOG_NOCARD_EVERY = 1.5   # s

# Edge nudge: if a scan starts and a card is detected near the frame
# edge, do a short pulsed rotation to bring it inward. This is not a
# capture gate; a card already inside the band is scanned immediately,
# and the nudge is time-capped.
EDGE_NUDGE_ENABLE   = True
EDGE_NUDGE_BAND     = 0.60
EDGE_NUDGE_SPEED    = 0.06   # rad/s
EDGE_NUDGE_ON       = 0.12   # s
EDGE_NUDGE_OFF      = 0.35   # s
EDGE_NUDGE_MAX_SECS = 4.0

# Visual band drawn on the live view (decoration only).
CENTER_BAND_FRAC = 0.70

CMD_VEL_TOPIC = "/cmd_vel"


# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
#
# A route is a list of (action, value) tuples:
#     ("DRIVE",  d)     drive d metres (negative reverses)
#     ("TURN",   a)     turn a degrees (positive is left)
#     ("SCAN",   None)  hold position and capture the card
#     ("BREAK",  None)  pause for BREAK_SECONDS while the cards are flipped
#     ("GOHOME", None)  return to the starting pose

def build_learn_route(n_cards):
    """Build the forward pass that learns all cards in order."""
    route = []
    for i in range(n_cards):
        route.append(("DRIVE", +STEP_DIST))
        route.append(("TURN",  +TURN_ANGLE))
        route.append(("SCAN",  None))
        if i < n_cards - 1:
            route.append(("TURN", -TURN_ANGLE))
    return route


def build_test_route(n_cards):
    """
    Build the return pass that tests each card.

    The robot is already facing the last learned card at the start. It
    turns away for the human break, turns back, then walks the row in
    reverse so that test scan #i corresponds to memory index N-1-i.
    """
    route = [
        ("TURN",  -TURN_ANGLE),   # turn away while cards are flipped
        ("BREAK", None),
        ("TURN",  +TURN_ANGLE),   # turn back to the last card
    ]
    for i in range(n_cards):
        route.append(("SCAN", None))
        if i < n_cards - 1:
            route.append(("TURN",  +TURN_ANGLE))
            route.append(("DRIVE", +STEP_DIST))
            route.append(("TURN",  -TURN_ANGLE))
    route.append(("GOHOME", None))
    return route


# -----------------------------------------------------------------------------
# Image enhancement
# -----------------------------------------------------------------------------

_GAMMA_LUT = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]
).astype("uint8")
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))


def enhance(bgr):
    """Denoise, then equalise contrast, gamma-correct, and sharpen."""
    if DENOISE_ENABLE:
        bgr = cv2.bilateralFilter(bgr, DENOISE_DIAMETER,
                                  DENOISE_SIGMA, DENOISE_SIGMA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    out = cv2.LUT(out, _GAMMA_LUT)
    blurred = cv2.GaussianBlur(out, (0, 0), UNSHARP_SIGMA)
    return cv2.addWeighted(out, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0)


# -----------------------------------------------------------------------------
# Vision helpers
# -----------------------------------------------------------------------------

def order_points(pts):
    """Sort four corners into TL, TR, BR, BL order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def get_warped_card(frame, pts, width=200, height=300):
    """Perspective-warp the four-corner region to a canonical card crop."""
    rect = order_points(pts)
    dst  = np.array([[0, 0], [width - 1, 0],
                     [width - 1, height - 1], [0, height - 1]],
                    dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (width, height))


def detect_color(warped_bgr):
    """Return 'Red' if the centre region contains red ink, else 'Black'."""
    pip = warped_bgr[55:245, 25:175]
    hsv = cv2.cvtColor(pip, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0,   60, 60]), np.array([10,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 255]))
    return "Red" if cv2.countNonZero(r1 + r2) > 100 else "Black"


def extract_pip_zone(warped_bgr):
    return warped_bgr[55:245, 25:175]


def laplacian_sharpness(gray):
    """Variance of the Laplacian; higher means sharper edges."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_biggest_card(frame_gray, debug=None, detail_gray=None):
    """
    Find the card's outer edge in a grayscale frame.

    The detector runs two thresholding methods (adaptive and Canny) and
    keeps every four-corner contour that passes the area, aspect, and
    fill gates. Contours nested inside another contour are rejected so
    that logos and pip boxes inside the card cannot win.

    Among the remaining candidates, the one whose bounding box contains
    the most other candidates is chosen. The card encloses its printed
    marks, so this rule favours the true outer edge.
    """
    H, W = frame_gray.shape[:2]
    blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Method A: adaptive threshold (the card is a bright blob).
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=51, C=-10)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                           np.ones((5, 5), np.uint8), iterations=2)

    # Method B: Canny edges (the card is an outline).
    edges = cv2.Canny(blur, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    stats = {"raw": 0, "quad": 0, "area_ok": 0, "aspect_ok": 0,
             "fill_ok": 0, "detail_ok": 0, "best_detail": 0.0}

    candidates = []
    for mask in (thr, edges):
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
        stats["raw"] += len(contours)
        if hier is None:
            continue
        hier = hier[0]
        for ci, c in enumerate(contours):
            # Skip nested contours (parent != -1 means inside another).
            if hier[ci][3] != -1:
                continue

            area = cv2.contourArea(c)
            if not (CARD_MIN_AREA < area < CARD_MAX_AREA):
                continue
            stats["area_ok"] += 1

            approx = None
            for eps in (0.02, 0.04, 0.06, 0.08):
                a = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)
                if len(a) == 4:
                    approx = a
                    break
            if approx is None:
                continue
            stats["quad"] += 1

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw == 0 or bh == 0:
                continue

            # A tilted card has the wrong axis-aligned aspect ratio, so
            # the shape gates use minAreaRect (the rotated bounding box).
            (_, (rw, rh), _) = cv2.minAreaRect(approx)
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / float(min(rw, rh))
            if not (CARD_ASPECT_MIN <= aspect <= CARD_ASPECT_MAX):
                continue
            stats["aspect_ok"] += 1

            fill = area / float(rw * rh)
            if fill < CARD_MIN_FILL:
                continue
            stats["fill_ok"] += 1

            if ENABLE_DETAIL_GATE:
                inset_x = int(bw * 0.18)
                inset_y = int(bh * 0.18)
                x0 = max(0, x + inset_x);   y0 = max(0, y + inset_y)
                x1 = min(W, x + bw - inset_x)
                y1 = min(H, y + bh - inset_y)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                detail_src = detail_gray if detail_gray is not None else frame_gray
                inner = detail_src[y0:y1, x0:x1]
                inner_edges = cv2.Canny(inner, 50, 150)
                detail = cv2.countNonZero(inner_edges) / float(inner.size)
                if detail > stats["best_detail"]:
                    stats["best_detail"] = detail
                if detail < CARD_MIN_DETAIL:
                    continue
            stats["detail_ok"] += 1

            candidates.append((approx, area, (x, y, bw, bh)))

    if debug is not None:
        debug.update(stats)

    if not candidates:
        return None, 0

    # Pick the candidate whose bounding box contains the most others;
    # break ties by area. This favours the true outer edge over inner
    # logos or pip boxes.
    def contains(outer, inner):
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (ix >= ox and iy >= oy and
                ix + iw <= ox + ow and iy + ih <= oy + oh)

    best, best_score = None, (-1, -1.0)
    for approx, area, bbox in candidates:
        enclosed = sum(1 for (_, _, ob) in candidates
                       if ob is not bbox and contains(bbox, ob))
        score = (enclosed, area)
        if score > best_score:
            best_score, best = score, (approx, area)

    return best


def estimate_rotation_angle(mem_gray, cur_gray):
    """
    Estimate the rotation between two card crops using ORB features.

    Features are matched between the memorised and current crop, then a
    RANSAC affine transform from mem to cur is fitted. The rotation
    angle of that transform comes out of atan2(M[1,0], M[0,0]).

    Returns (angle_degrees, n_matches, matched). When matched is False
    the angle is meaningless and the caller should treat the card as
    unchanged.
    """
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(mem_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None:
        return 0.0, 0, False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return 0.0, len(matches), False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return 0.0, len(matches), False

    angle_rad = math.atan2(M[1, 0], M[0, 0])
    angle_deg = math.degrees(angle_rad) % 360
    return angle_deg, len(matches), True


def find_flip_for_card(mem_gray, cur_gray):
    """
    Decide whether the current card was flipped 180 degrees relative to
    its memorised version. Returns (is_flip, info) where info carries the
    diagnostic numbers for logging.
    """
    angle, n_matches, matched = estimate_rotation_angle(mem_gray, cur_gray)

    if matched:
        is_flip = (140.0 < angle < 220.0)
    else:
        is_flip = False

    info = {
        "angle":     angle,
        "n_matches": n_matches,
        "matched":   matched,
    }
    return is_flip, info


def build_memory_grid(pip_list, labels, flipped=None):
    """
    Compose a horizontal strip of card thumbnails with labels. Tiles whose
    flipped flag is True are drawn with a red border and label.
    """
    TW, TH = 150, 190
    n = len(pip_list)
    if n == 0:
        return None
    if flipped is None:
        flipped = [False] * n

    grid = np.ones((TH + 30, TW * n, 3), dtype=np.uint8) * 40
    for i, pip in enumerate(pip_list):
        tile = cv2.resize(pip, (TW, TH))
        grid[:TH, i*TW:(i+1)*TW] = tile

        is_flipped = i < len(flipped) and flipped[i]
        if is_flipped:
            cv2.rectangle(grid, (i*TW + 1, 1),
                          ((i+1)*TW - 2, TH + 28), (0, 0, 255), 4)
            label_col = (0, 0, 255)
            text = labels[i] + "  FLIPPED"
        else:
            label_col = (0, 255, 255)
            text = labels[i]

        cv2.putText(grid, text, (i*TW + 6, TH + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_col, 1)
    return grid


def yaw_from_quaternion(q):
    """Yaw angle in radians from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# -----------------------------------------------------------------------------
# ROS 2 node
# -----------------------------------------------------------------------------

class CardScannerRoute(Node):

    def __init__(self):
        super().__init__('card_scanner_route')
        self.bridge = CvBridge()

        # Memory of learned cards.
        self.memory_gray    = []
        self.memory_pip_bgr = []
        self.memory_labels  = []
        self.memory_flipped = []
        self.tested_cards   = 0

        # Latest perception state, updated by image_callback.
        self.latest_card_gray  = None
        self.latest_card_bgr   = None
        self.scan_good_frames  = 0
        self.scan_miss_count   = 0
        self.last_sharpness    = 0.0
        self.last_card_present = False
        self._last_nocard_log  = 0.0
        self._frame_count      = 0
        self.card_cx_frac      = 0.5
        self.card_partial      = False
        self._nudge_pulse_t0   = 0.0
        self._nudge_start_t    = 0.0
        self.last_motion_time  = 0.0
        self._ros_wall_offset  = None

        # Odometry.
        self.have_odom = False
        self.odom_x    = 0.0
        self.odom_y    = 0.0
        self.odom_yaw  = 0.0

        # Route: the learn pass followed by the test pass.
        self.route = build_learn_route(TARGET_CARDS) \
                   + build_test_route(TARGET_CARDS)
        self.route_idx  = 0
        self.step_phase = "START"
        self.step_x0    = 0.0
        self.step_y0    = 0.0
        self.step_yaw0  = 0.0
        self.step_t0    = 0.0

        # Home pose, latched from the first odometry reading. The robot
        # returns here at the end of the run.
        self.home_x      = None
        self.home_y      = 0.0
        self.home_yaw    = 0.0
        self.target_yaw  = 0.0
        self.gohome_phase = "TURN_TO"

        self.display_text = "Waiting for odometry..."

        # Publishers and subscribers.
        self.vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/image_raw',
                                 self.image_callback, video_qos)
        self.create_subscription(Odometry, '/odom',
                                 self.odom_callback, 10)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            "Card scanner started. Live view in OpenCV popup windows.")
        self.get_logger().info(
            f"Route has {len(self.route)} steps for {TARGET_CARDS} cards.")

    # ---- Callbacks --------------------------------------------------------

    def odom_callback(self, msg):
        self.have_odom = True
        self.odom_x   = msg.pose.pose.position.x
        self.odom_y   = msg.pose.pose.position.y
        self.odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.home_x is None:
            self.home_x     = self.odom_x
            self.home_y     = self.odom_y
            self.home_yaw   = self.odom_yaw
            self.target_yaw = self.odom_yaw
            self.get_logger().info(
                f"HOME pose latched: x={self.home_x:.2f} "
                f"y={self.home_y:.2f} yaw={math.degrees(self.home_yaw):.0f}")

            # Warn if /odom looks stale (a fresh bringup starts near zero).
            drift = math.hypot(self.home_x, self.home_y)
            if drift > 0.20 or abs(self.home_yaw) > math.radians(20):
                self.get_logger().warn(
                    f"Odometry looks stale: home pose is {drift:.2f} m / "
                    f"{math.degrees(self.home_yaw):.0f} deg from origin. "
                    "Restart the robot bringup before running again.")

    def stop_robot(self):
        self.vel_pub.publish(Twist())

    def drive(self, linear=0.0, angular=0.0):
        t = Twist()
        t.linear.x  = float(linear)
        t.angular.z = float(angular)
        self.vel_pub.publish(t)
        if abs(linear) > 1e-6 or abs(angular) > 1e-6:
            self.last_motion_time = time.time()

    def dist_travelled(self):
        return math.hypot(self.odom_x - self.step_x0,
                          self.odom_y - self.step_y0)

    # ---- Main control loop (20 Hz) ---------------------------------------

    def control_loop(self):
        if not self.have_odom:
            self.display_text = "Waiting for /odom..."
            self.stop_robot()
            return

        if self.route_idx >= len(self.route):
            self.display_text = "ROUTE COMPLETE"
            self.stop_robot()
            return

        action, value = self.route[self.route_idx]

        # On entering a step, latch the reference pose and time.
        if self.step_phase == "START":
            self.step_x0   = self.odom_x
            self.step_y0   = self.odom_y
            self.step_yaw0 = self.odom_yaw
            self.step_t0   = time.time()
            self._nudge_pulse_t0 = time.time()
            self._nudge_start_t  = 0.0
            self.gohome_phase    = "TURN_TO"

            # Turns target an absolute heading rather than a relative one.
            # Any error left over from a previous turn is corrected here
            # instead of being carried forward.
            if action == "TURN":
                self.target_yaw = math.atan2(
                    math.sin(self.target_yaw + math.radians(value)),
                    math.cos(self.target_yaw + math.radians(value)))

            self.step_phase = "RUNNING"
            self.get_logger().info(
                f"[step {self.route_idx + 1}/{len(self.route)}] "
                f"{action} {value}")

        if action == "DRIVE":
            target    = abs(value)
            direction = 1.0 if value >= 0 else -1.0
            self.display_text = (f"DRIVE {value:+.2f} m  "
                                 f"({self.dist_travelled():.2f})")
            if self.dist_travelled() >= target - DIST_TOL:
                self.stop_robot()
                self._next_step()
            else:
                self.drive(linear=direction * DRIVE_SPEED)

        elif action == "TURN":
            err = math.atan2(math.sin(self.target_yaw - self.odom_yaw),
                             math.cos(self.target_yaw - self.odom_yaw))
            self.display_text = (f"TURN {value:+.0f}\u00b0  "
                                 f"(err {math.degrees(err):+.0f})")
            if abs(err) <= ANGLE_TOL:
                self.stop_robot()
                self._next_step()
            else:
                # Slow down near the target to avoid overshoot.
                if abs(err) >= TURN_SLOW_ZONE:
                    speed = TURN_SPEED
                else:
                    speed = max(TURN_MIN_SPEED,
                                TURN_SPEED * abs(err) / TURN_SLOW_ZONE)
                self.drive(angular=math.copysign(speed, err))

        elif action == "SCAN":
            self.stop_robot()
            elapsed = time.time() - self.step_t0

            if elapsed < SCAN_SETTLE_SECS:
                self.display_text = (f"SCANNING... settle "
                                     f"{SCAN_SETTLE_SECS - elapsed:.1f}s")
                self.scan_good_frames = 0
                return

            # If a card is visible but sits near the frame edge, nudge
            # the robot a little so the next capture is more central.
            if EDGE_NUDGE_ENABLE and self.last_card_present:
                offset = self.card_cx_frac - 0.5
                half_band = EDGE_NUDGE_BAND / 2.0
                budget_left = (
                    self._nudge_start_t == 0.0 or
                    time.time() - self._nudge_start_t < EDGE_NUDGE_MAX_SECS)
                if abs(offset) > half_band and budget_left:
                    if self._nudge_start_t == 0.0:
                        self._nudge_start_t  = time.time()
                        self._nudge_pulse_t0 = time.time()
                    direction = -1.0 if offset > 0 else +1.0
                    side = "RIGHT" if offset > 0 else "LEFT"
                    cycle = EDGE_NUDGE_ON + EDGE_NUDGE_OFF
                    phase = (time.time() - self._nudge_pulse_t0) % cycle
                    self.display_text = (f"NUDGE card in ({side})  "
                                         f"x={self.card_cx_frac:.2f}")
                    if phase < EDGE_NUDGE_ON:
                        self.drive(angular=direction * EDGE_NUDGE_SPEED)
                    else:
                        self.stop_robot()
                    self.scan_good_frames = 0
                    return

            if self.scan_good_frames >= SCAN_GOOD_FRAMES:
                self.get_logger().info(
                    f"SCAN: captured (sharpness={self.last_sharpness:.0f})")
                self._do_scan()
                self._next_step()
            else:
                if self.last_card_present:
                    self.stop_robot()
                    self.display_text = (
                        f"SCANNING... waiting sharp frame "
                        f"({self.scan_good_frames}/{SCAN_GOOD_FRAMES})  "
                        f"sharp={self.last_sharpness:.0f}")
                else:
                    self.stop_robot()
                    self.display_text = ("NO CARD - waiting. "
                                         "Place a card under the camera.")
                    if time.time() - self._last_nocard_log > 5.0:
                        self._last_nocard_log = time.time()
                        self.get_logger().warn(
                            "SCAN: no card detected. Holding position.")

        elif action == "BREAK":
            self.stop_robot()
            elapsed   = time.time() - self.step_t0
            remaining = max(0, BREAK_SECONDS - elapsed)
            self.display_text = f"FLIP CARDS NOW - {int(remaining)}s"
            if elapsed >= BREAK_SECONDS:
                self.get_logger().info("Break over. Testing pass begins.")
                self._next_step()

        elif action == "GOHOME":
            self._do_gohome()

    def _do_gohome(self):
        """Three-phase return to the latched home pose."""
        if self.home_x is None:
            self._next_step()
            return

        dx = self.home_x - self.odom_x
        dy = self.home_y - self.odom_y
        dist_home = math.hypot(dx, dy)

        if self.gohome_phase == "TURN_TO":
            if dist_home < DIST_TOL:
                self.gohome_phase = "TURN_FINAL"
                return
            heading = math.atan2(dy, dx)
            err = math.atan2(math.sin(heading - self.odom_yaw),
                             math.cos(heading - self.odom_yaw))
            self.display_text = (f"GO HOME: aiming  "
                                 f"({math.degrees(err):+.0f}\u00b0)")
            if abs(err) <= ANGLE_TOL:
                self.stop_robot()
                self.gohome_phase = "DRIVE"
            else:
                if abs(err) >= TURN_SLOW_ZONE:
                    speed = TURN_SPEED
                else:
                    speed = max(TURN_MIN_SPEED,
                                TURN_SPEED * abs(err) / TURN_SLOW_ZONE)
                self.drive(angular=math.copysign(speed, err))

        elif self.gohome_phase == "DRIVE":
            self.display_text = f"GO HOME: driving  ({dist_home:.2f} m)"
            if dist_home <= DIST_TOL:
                self.stop_robot()
                self.gohome_phase = "TURN_FINAL"
            else:
                heading = math.atan2(dy, dx)
                err = math.atan2(math.sin(heading - self.odom_yaw),
                                 math.cos(heading - self.odom_yaw))
                self.drive(linear=DRIVE_SPEED,
                           angular=max(-0.3, min(0.3, 1.5 * err)))

        elif self.gohome_phase == "TURN_FINAL":
            err = math.atan2(math.sin(self.home_yaw - self.odom_yaw),
                             math.cos(self.home_yaw - self.odom_yaw))
            self.display_text = (f"GO HOME: final turn  "
                                 f"({math.degrees(err):+.0f}\u00b0)")
            if abs(err) <= ANGLE_TOL:
                self.stop_robot()
                self.get_logger().info("GO HOME complete.")
                self._next_step()
            else:
                if abs(err) >= TURN_SLOW_ZONE:
                    speed = TURN_SPEED
                else:
                    speed = max(TURN_MIN_SPEED,
                                TURN_SPEED * abs(err) / TURN_SLOW_ZONE)
                self.drive(angular=math.copysign(speed, err))

    def _next_step(self):
        self.route_idx += 1
        self.step_phase = "START"

    # ---- Scan action: learn or test --------------------------------------

    def _do_scan(self):
        if self.latest_card_gray is None:
            self.get_logger().warn("SCAN: no card under the camera.")
            return

        learning = len(self.memory_gray) < TARGET_CARDS
        if learning:
            gray  = self.latest_card_gray
            bgr   = self.latest_card_bgr
            color = detect_color(bgr)
            n     = len(self.memory_gray) + 1
            label = f"C{n} {color[0]}"
            self.memory_gray.append(gray)
            self.memory_pip_bgr.append(extract_pip_zone(bgr))
            self.memory_labels.append(label)
            self.memory_flipped.append(False)
            self._show_memory_popup(build_memory_grid(
                self.memory_pip_bgr, self.memory_labels, self.memory_flipped))
            self.get_logger().info(f"LEARNED: {label} (Card {n})")
        else:
            cur = self.latest_card_gray
            # Test scan #i corresponds to memory index N-1-i, because the
            # robot walks the row in reverse on the return pass.
            idx = (TARGET_CARDS - 1) - self.tested_cards
            self.tested_cards += 1
            if idx < 0 or idx >= len(self.memory_gray):
                self.get_logger().warn(
                    f"TEST: position index {idx} out of range.")
                return

            is_flip, info = find_flip_for_card(self.memory_gray[idx], cur)
            lbl     = self.memory_labels[idx]
            verdict = "FLIPPED!" if is_flip else "untouched"
            note    = "  [partial capture]" if self.card_partial else ""
            self.get_logger().info(
                f"TEST RESULT: {lbl} -> {verdict}  "
                f"[angle={info['angle']:.0f}\u00b0 "
                f"matches={info['n_matches']} "
                f"matched={info['matched']}]{note}")
            self.memory_flipped[idx] = bool(is_flip)
            self._show_memory_popup(build_memory_grid(
                self.memory_pip_bgr, self.memory_labels, self.memory_flipped))

    # ---- Popup windows ---------------------------------------------------

    def _show_popup(self, frame):
        """Show the live frame, throttled so the GUI thread stays responsive."""
        if not POPUP_ENABLE or getattr(self, "_popup_broken", False):
            return
        self._popup_skip = getattr(self, "_popup_skip", 0) + 1
        if self._popup_skip % POPUP_EVERY_NTH != 0:
            return
        try:
            cv2.imshow("Card Scanner (popup)", frame)
            cv2.waitKey(1)
        except Exception as e:
            self._popup_broken = True
            self.get_logger().warn(
                f"Popup disabled: no display available ({e}).")

    def _show_memory_popup(self, grid):
        """Show the learned-card grid; updated only when memory changes."""
        if not POPUP_ENABLE or getattr(self, "_popup_broken", False):
            return
        if grid is None:
            return
        try:
            cv2.imshow("Card Scanner (memory)", grid)
            cv2.waitKey(1)
        except Exception:
            pass

    # ---- Image callback --------------------------------------------------

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        # The image header carries the capture time. The offset between
        # ROS time and wall time is measured once and then used to reject
        # stale frames during a scan.
        try:
            frame_capture_t = (msg.header.stamp.sec +
                               msg.header.stamp.nanosec * 1e-9)
        except Exception:
            frame_capture_t = 0.0
        if self._ros_wall_offset is None and frame_capture_t > 0:
            self._ros_wall_offset = time.time() - frame_capture_t
        frame_wall_t = (frame_capture_t + self._ros_wall_offset
                        if self._ros_wall_offset is not None
                        else time.time())

        action = (self.route[self.route_idx][0]
                  if self.route_idx < len(self.route) else "DONE")
        heavy = (action == "SCAN")

        # Outside of scans the enhance + detect pipeline is wasted work,
        # so the callback takes a cheap path that only resizes and shows.
        if not heavy:
            self.latest_card_gray  = None
            self.latest_card_bgr   = None
            self.last_card_present = False
            self.card_partial      = False
            self.scan_good_frames  = 0
            self.scan_miss_count   = 0
            small = cv2.resize(frame, (STREAM_W, STREAM_H))
            cv2.putText(small, self.display_text, (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._show_popup(small)
            return

        # Stale frames captured before the last motion ended are dropped
        # from the detection path, although they still appear on the popup.
        frame_is_stale = (self.last_motion_time > 0.0 and
                          frame_wall_t < self.last_motion_time
                          + FRAME_FRESH_GUARD)
        if frame_is_stale:
            hh = frame.shape[0]
            light = cv2.resize(frame[hh // 2:, :], (STREAM_W, STREAM_H))
            cv2.putText(light, "waiting for fresh frame...", (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            self._show_popup(light)
            return

        # Only run the heavy enhance + detect path on every Nth frame.
        self._frame_count += 1
        if self._frame_count % SCAN_PROCESS_EVERY != 0:
            hh = frame.shape[0]
            light = cv2.resize(frame[hh // 2:, :], (STREAM_W, STREAM_H))
            cv2.putText(light, self.display_text, (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self._show_popup(light)
            return

        frame = enhance(frame)
        h, w  = frame.shape[:2]
        frame = frame[h // 2:, :]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blank the top band so that the wall / skirting edge cannot
        # produce a contour that competes with the card.
        ch = gray.shape[0]
        wall_y = int(ch * DETECT_TOP_IGNORE)
        gray_roi = gray.copy()
        gray_roi[:wall_y, :] = 127

        # An alternative grayscale where every channel's max value wins.
        # Used only by the optional detail gate; the outline detection
        # always uses the standard grayscale.
        detail_gray_full = np.max(frame, axis=2).astype(np.uint8)
        detail_gray_full[:wall_y, :] = 127

        dbg = {}
        approx, area = detect_biggest_card(gray_roi, debug=dbg,
                                           detail_gray=detail_gray_full)

        # Mark the wall band and the central guide band on the live view.
        cv2.line(frame, (0, wall_y), (frame.shape[1], wall_y),
                 (90, 90, 90), 1)
        fw = frame.shape[1]
        fh = frame.shape[0]
        side_f  = (1.0 - CENTER_BAND_FRAC) / 2.0
        x_left  = int(fw * side_f)
        x_right = int(fw * (1.0 - side_f))
        cv2.line(frame, (x_left, 0),  (x_left, fh),  (0, 200, 255), 2)
        cv2.line(frame, (x_right, 0), (x_right, fh), (0, 200, 255), 2)

        if approx is not None:
            pts = approx.reshape(4, 2)
            self.latest_card_gray  = get_warped_card(gray, pts)
            self.latest_card_bgr   = get_warped_card(frame, pts)
            self.last_card_present = True

            bx, by, bw_, bh_ = cv2.boundingRect(approx)
            self.card_cx_frac = (bx + bw_ / 2.0) / float(fw)

            # If the card box touches a frame edge the capture is only
            # partial. Detection still works but the flip verdict will
            # be less reliable.
            edge = 4
            self.card_partial = (bx <= edge or by <= edge or
                                 bx + bw_ >= fw - edge or
                                 by + bh_ >= fh - edge)

            sharp = laplacian_sharpness(self.latest_card_gray)
            self.last_sharpness = sharp

            # A blurred-but-present frame holds the streak rather than
            # resetting it, so brief sensor noise doesn't undo progress.
            if sharp >= SHARPNESS_MIN:
                self.scan_good_frames += 1
                self.scan_miss_count = 0

            box_col = (0, 165, 255) if self.card_partial else (0, 255, 0)
            cv2.drawContours(frame, [approx], -1, box_col, 3)
            tag = "  PARTIAL" if self.card_partial else ""
            cv2.putText(frame,
                        f"CARD area={int(area)} sharp={int(sharp)}{tag}",
                        (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        box_col, 2)
        else:
            self.latest_card_gray  = None
            self.latest_card_bgr   = None
            self.last_card_present = False
            self.card_partial      = False

            # A single miss is forgiven; the streak only collapses after
            # a few misses in a row.
            self.scan_miss_count += 1
            if self.scan_miss_count >= SCAN_MISS_TOLERANCE:
                self.scan_good_frames = 0

            reason = (f"NO CARD  raw={dbg.get('raw',0)} "
                      f"areaOK={dbg.get('area_ok',0)} "
                      f"quad={dbg.get('quad',0)} "
                      f"aspectOK={dbg.get('aspect_ok',0)} "
                      f"fillOK={dbg.get('fill_ok',0)} "
                      f"detailOK={dbg.get('detail_ok',0)}")
            cv2.putText(frame, reason, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            now = time.time()
            if now - self._last_nocard_log > DEBUG_LOG_NOCARD_EVERY:
                self._last_nocard_log = now
                self.get_logger().warn(reason)

        cv2.putText(frame, self.display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"Learned {len(self.memory_gray)}/{TARGET_CARDS}    "
                    f"Tested {self.tested_cards}/{TARGET_CARDS}",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        final = cv2.resize(frame, (STREAM_W, STREAM_H))
        self._show_popup(final)


def main(args=None):
    rclpy.init(args=args)
    node = CardScannerRoute()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl+C the rclpy context may already be torn down, so each
        # cleanup step is guarded individually.
        try:
            node.stop_robot()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
