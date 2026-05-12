import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist  # NEW: For moving the wheels!
from cv_bridge import CvBridge
import cv2
import numpy as np
import pytesseract
import math
import time # NEW: For timers

# --- Helper Functions ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_warped_card(frame, pts, width=200, height=300):
    rect = order_points(pts)
    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (width, height))

def read_card_identity(warped_bgr):
    corner = warped_bgr[5:85, 5:45]
    hsv_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_corner, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_corner, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    color_str = "Red" if cv2.countNonZero(red_mask) > 50 else "Black"

    gray_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_corner, 150, 255, cv2.THRESH_BINARY_INV)
    custom_config = r'--psm 10 -c tessedit_char_whitelist=2345678910JQKA'
    rank = pytesseract.image_to_string(thresh, config=custom_config).strip()
    if not rank: rank = "?"
    return f"{rank} of {color_str}"

def check_flip_angle(mem_card_gray, cur_card_gray):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(mem_card_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_card_gray, None)
    if des1 is None or des2 is None: return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10: return 0.0
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None: return 0.0
        
    angle_rad = math.atan2(M[1, 0], M[0, 0])
    return (math.degrees(angle_rad) % 360)


# --- ROS 2 NODE ---
class CardMemoryNode(Node):
    def __init__(self):
        super().__init__('card_memory_node')
        self.bridge = CvBridge()
        
        # Memory variables
        self.memory_state_gray = []
        self.memory_identities = []
        self.current_cards_gray = []
        self.current_cards_bgr = []
        self.flipped_card_index = -1
        
        # Subscribers and Publishers
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # Publishes wheel commands
        
        # State Machine Variables
        self.state = "WAITING"
        self.state_start_time = time.time()
        self.stable_frames = 0
        self.target_cards = 5
        
        # --- TUNING VARIABLES ---
        self.turn_speed = 0.5      
        self.turn_away_duration = 6.6  # Shaving off another quarter-second!
        self.turn_back_duration = 6.28 
        self.hide_duration = 5.0
        
        # Control Loop Timer (Runs 10 times a second)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Autonomous Mode Started! Place 5 cards in front of the bot.")

    def change_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f"--- STATE CHANGE -> {new_state} ---")

    def control_loop(self):
        """The internal brain clock that manages the state machine."""
        elapsed = time.time() - self.state_start_time
        vel_cmd = Twist()

        if self.state == "WAITING":
            # Wait for 5 cards to be stable for at least 15 frames (~1.5 seconds)
            if len(self.current_cards_gray) == self.target_cards:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
                
            if self.stable_frames >= 15:
                self.change_state("SAVING")

        elif self.state == "SAVING":
            # Save the memory automatically
            self.memory_state_gray = self.current_cards_gray.copy()
            self.memory_identities = [read_card_identity(c) for c in self.current_cards_bgr]
            self.get_logger().info("Memory locked! Turning away...")
            self.change_state("TURNING_AWAY")

        elif self.state == "TURNING_AWAY":
            if elapsed < self.turn_away_duration: # <-- CHANGED THIS LINE
                vel_cmd.angular.z = self.turn_speed 
            else:
                self.change_state("HIDING")
            self.vel_pub.publish(vel_cmd)

        elif self.state == "HIDING":
            self.vel_pub.publish(Twist()) # Force stop
            # Wait 5 seconds for human to flip a card
            if elapsed >= self.hide_duration:
                self.get_logger().info("Time is up! Turning back...")
                self.change_state("TURNING_BACK")

        elif self.state == "TURNING_BACK":
            if elapsed < self.turn_back_duration: # <-- CHANGED THIS LINE
                vel_cmd.angular.z = -self.turn_speed 
            else:
                self.stable_frames = 0
                self.change_state("SETTLING")
            self.vel_pub.publish(vel_cmd)

        elif self.state == "SETTLING":
            self.vel_pub.publish(Twist()) # Force stop
            # Wait until it clearly sees 5 cards again
            if len(self.current_cards_gray) == self.target_cards:
                self.stable_frames += 1
            if self.stable_frames >= 15:
                self.change_state("COMPARING")
            # Failsafe: If it doesn't see 5 cards after 10 seconds, complain.
            elif elapsed > 10.0:
                self.get_logger().warn("I don't see 5 cards anymore! Move them back into view!")

        elif self.state == "COMPARING":
            self.get_logger().info("[RESULTS]:")
            self.flipped_card_index = -1 # Reset before checking
            
            for i in range(len(self.memory_state_gray)):
                mem_gray = self.memory_state_gray[i]
                cur_gray = self.current_cards_gray[i]
                identity = self.memory_identities[i]
                angle = check_flip_angle(mem_gray, cur_gray)
                
                if 140 < angle < 220:
                    self.get_logger().info(f" -> {identity} (Card {i+1}) was FLIPPED!")
                    self.flipped_card_index = i  # <--- NEW: Save the winning card number!
                else:
                    self.get_logger().info(f" -> {identity} (Card {i+1}) is untouched.")
            self.change_state("FINISHED")
            
        elif self.state == "FINISHED":
            # Do nothing, game is over!
            pass

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_contours = [cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True) 
                         for c in contours if cv2.contourArea(c) > 1500]
        card_contours = [c for c in card_contours if len(c) == 4]
        
        card_contours = sorted(card_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        temp_gray_cards = []
        temp_bgr_cards = []
        card_count = len(card_contours)
        
        for i, approx in enumerate(card_contours):
            # Default to Green for all cards
            box_color = (0, 255, 0)
            
            # NEW: If the game is over and this is the flipped card, make it RED!
            if self.state in ["COMPARING", "FINISHED"] and i == self.flipped_card_index:
                box_color = (0, 0, 255) # Red in BGR
                
            # Draw the box using our chosen color
            cv2.drawContours(frame, [approx], -1, box_color, 3)
            pts = approx.reshape(4, 2)
            
            temp_gray_cards.append(get_warped_card(gray, pts))
            temp_bgr_cards.append(get_warped_card(frame, pts))
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, f"Card {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        self.current_cards_gray = temp_gray_cards
        self.current_cards_bgr = temp_bgr_cards
        
        # Display the State Machine status on screen
        cv2.putText(frame, f"State: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.putText(frame, f"Cards Seen: {card_count}/5", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("TurtleBot3 Autonomous Vision", display_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CardMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vel_pub.publish(Twist()) # Send a stop command just in case!
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()