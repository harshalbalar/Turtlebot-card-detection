import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pytesseract
import math
import time

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
    corner = warped_bgr[5:95, 5:55]
    hsv_corner = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_corner, lower_red1, upper_red1) + cv2.inRange(hsv_corner, lower_red2, upper_red2)
    color_str = "Red" if cv2.countNonZero(red_mask) > 100 else "Black"

    number_only_crop = corner[0:60, 0:45]
    gray_corner = cv2.cvtColor(number_only_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_corner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.resize(thresh, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    custom_config = r'--psm 8 -c tessedit_char_whitelist=2345678910JQKA'
    rank = pytesseract.image_to_string(thresh, config=custom_config).strip()
    if len(rank) > 2 and rank != "10": rank = rank[0] 
    if not rank: rank = "?"
    return f"{rank} of {color_str}"

# NEW: Supercharged Memory Search Engine
def find_best_match_and_angle(memory_list, cur_card_gray):
    best_match_idx = -1
    best_inliers = 0
    best_angle = 0.0
    
    orb = cv2.ORB_create(nfeatures=500)
    kp2, des2 = orb.detectAndCompute(cur_card_gray, None)
    if des2 is None: return -1, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for i, mem_card in enumerate(memory_list):
        kp1, des1 = orb.detectAndCompute(mem_card, None)
        if des1 is None: continue
        
        matches = bf.match(des1, des2)
        if len(matches) < 10: continue
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is not None and inliers is not None:
            inlier_count = len(inliers)
            # If this memory has the most matching dots, it's our card!
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_match_idx = i
                angle_rad = math.atan2(M[1, 0], M[0, 0])
                best_angle = (math.degrees(angle_rad) % 360)

    return best_match_idx, best_angle

# --- ROS 2 NODE ---
class CardMemoryNode(Node):
    def __init__(self):
        super().__init__('card_memory_node')
        self.bridge = CvBridge()
        
        self.memory_state_gray = []
        self.memory_identities = []
        self.tested_cards = 0
        
        self.current_cards_gray = []
        self.current_cards_bgr = []
        
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile_sensor_data)
        
        # New Sequential State Machine
        self.state = "LEARNING"
        self.state_start_time = time.time()
        self.stable_frames = 0
        self.empty_frames = 0
        
        # --- NEW DISPLAY VARIABLES ---
        self.display_text = "Analyzing..."
        self.display_color = (255, 255, 0) # Default to Cyan
        # -----------------------------

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Sequential Mode Started! Show me Card 1.")

    def change_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f"--- STATE CHANGE -> {new_state} ---")

    def control_loop(self):
        elapsed = time.time() - self.state_start_time

        # PHASE 1: Scan exactly 1 card
        if self.state == "LEARNING":
            if len(self.current_cards_gray) == 1:
                self.stable_frames += 1
                if self.stable_frames >= 15: # Waited long enough to focus
                    card_gray = self.current_cards_gray[0]
                    card_bgr = self.current_cards_bgr[0]
                    
                    self.memory_state_gray.append(card_gray)
                    identity = read_card_identity(card_bgr)
                    self.memory_identities.append(identity)
                    
                    self.get_logger().info(f"SAVED: {identity} as Card {len(self.memory_state_gray)}")
                    self.stable_frames = 0
                    
                    if len(self.memory_state_gray) == 5:
                        self.change_state("BREAK")
                    else:
                        self.get_logger().info("Please REMOVE the card from the camera.")
                        self.change_state("WAIT_FOR_EMPTY_LEARN")
            else:
                self.stable_frames = 0

        # Wait for the human to pick the card up off the table
        elif self.state == "WAIT_FOR_EMPTY_LEARN":
            if len(self.current_cards_gray) == 0:
                self.empty_frames += 1
                if self.empty_frames >= 10:
                    self.empty_frames = 0
                    self.get_logger().info(f"Ready for Card {len(self.memory_state_gray) + 1}!")
                    self.change_state("LEARNING")
            else:
                self.empty_frames = 0

        # PHASE 2: Wait 5 seconds
        elif self.state == "BREAK":
            if elapsed >= 5.0:
                self.get_logger().info("Break is over! Show me the cards one by one (any order!).")
                self.change_state("TESTING")

        # PHASE 3: Show cards to check for flips
        elif self.state == "TESTING":
            if len(self.current_cards_gray) == 1:
                self.stable_frames += 1
                if self.stable_frames >= 15:
                    cur_gray = self.current_cards_gray[0]
                    
                    # Search the memory bank!
                    match_idx, angle = find_best_match_and_angle(self.memory_state_gray, cur_gray)
                    
                    if match_idx != -1:
                        identity = self.memory_identities[match_idx]
                        if 140 < angle < 220:
                            self.get_logger().info(f"RESULTS: That is {identity} (Original Card {match_idx+1}) AND IT IS FLIPPED!")
                            # NEW: Make it Red and shout Flipped!
                            self.display_text = f"FLIPPED: {identity}"
                            self.display_color = (0, 0, 255) 
                        else:
                            self.get_logger().info(f"RESULTS: That is {identity} (Original Card {match_idx+1}) (Untouched).")
                            # NEW: Keep it Cyan and say OK
                            self.display_text = f"OK: {identity}"
                            self.display_color = (255, 255, 0) 
                    else:
                        self.get_logger().warn("I don't remember this card at all!")
                        self.display_text = "UNKNOWN CARD"
                        self.display_color = (0, 165, 255) # Orange

                    self.tested_cards += 1
                    self.stable_frames = 0
                    
                    if self.tested_cards == 5:
                        self.change_state("FINISHED")
                    else:
                        self.get_logger().info("Please REMOVE the card from the camera.")
                        self.change_state("WAIT_FOR_EMPTY_TEST")
            else:
                self.stable_frames = 0

        elif self.state == "WAIT_FOR_EMPTY_TEST":
            if len(self.current_cards_gray) == 0:
                self.empty_frames += 1
                if self.empty_frames >= 10:
                    self.empty_frames = 0
                    self.display_text = "Analyzing..." # <--- NEW: Reset text for next card
                    self.display_color = (255, 255, 0) # <--- NEW: Reset color for next card
            else:
                self.empty_frames = 0

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
                         for c in contours if 1500 < cv2.contourArea(c) < 40000]
        card_contours = [c for c in card_contours if len(c) == 4]
        
        card_contours = sorted(card_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        temp_gray_cards = []
        temp_bgr_cards = []
        
        for i, approx in enumerate(card_contours):
            # Dynamic colors based on what the bot is doing
            if self.state in ["LEARNING", "WAIT_FOR_EMPTY_LEARN"]:
                box_color = (0, 255, 0) # Green
                text_to_show = "Scanning..."
            else:
                # Use the variables we set during the Testing phase!
                box_color = self.display_color
                text_to_show = self.display_text
                
            cv2.drawContours(frame, [approx], -1, box_color, 3)
            pts = approx.reshape(4, 2)
            
            temp_gray_cards.append(get_warped_card(gray, pts))
            temp_bgr_cards.append(get_warped_card(frame, pts))
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, text_to_show, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
        self.current_cards_gray = temp_gray_cards
        self.current_cards_bgr = temp_bgr_cards
        
        cv2.putText(frame, f"State: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, f"Cards Memorized: {len(self.memory_state_gray)}/5", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if self.state in ["TESTING", "WAIT_FOR_EMPTY_TEST", "FINISHED"]:
            cv2.putText(frame, f"Cards Tested: {self.tested_cards}/5", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("TurtleBot3 Magic Scanner", display_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CardMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
