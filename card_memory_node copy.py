import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import pytesseract
import math

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
        self.first_frame_received = False
        
        self.memory_state_gray = []
        self.memory_identities = []
        self.current_cards_gray = []
        self.current_cards_bgr = []
        
        # --- FIXED: Changed back to 10 for "Reliable" QoS ---
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
            
        self.command_sub = self.create_subscription(
            String, '/card_memory/command', self.command_callback, 10)
            
        self.image_pub = self.create_publisher(Image, '/card_memory/debug_image', 10)
        
        self.get_logger().info("TurtleBot3 Card Memory Node Started!")
        self.get_logger().info("Waiting for the camera feed to connect...")

    def image_callback(self, msg):
        if not self.first_frame_received:
            self.get_logger().info("✅ SUCCESS: Receiving camera video feed!")
            self.first_frame_received = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
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
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            pts = approx.reshape(4, 2)
            
            temp_gray_cards.append(get_warped_card(gray, pts))
            temp_bgr_cards.append(get_warped_card(frame, pts))
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, f"Card {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.current_cards_gray = temp_gray_cards
        self.current_cards_bgr = temp_bgr_cards
        
        cv2.putText(frame, f"Total Cards Seen: {card_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    
        if len(self.memory_state_gray) > 0:
            cv2.putText(frame, "[MEMORY ACTIVE]", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Shrink the frame by 50% strictly for the popup window
        display_frame = cv2.resize(frame, (640, 480))
        cv2.imshow("TurtleBot3 Live Vision", display_frame)
        cv2.waitKey(1)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(debug_msg)
        except Exception:
            pass

    def command_callback(self, msg):
        cmd = msg.data.lower()
        if cmd == 'save':
            if not self.current_cards_gray:
                self.get_logger().warn("No cards currently detected to save!")
                return
                
            self.memory_state_gray = self.current_cards_gray.copy()
            self.memory_identities = []
            self.get_logger().info(f"[MEMORY SAVED] Memorized {len(self.memory_state_gray)} cards.")
            
            for bgr_card in self.current_cards_bgr:
                identity = read_card_identity(bgr_card)
                self.memory_identities.append(identity)
                self.get_logger().info(f" - {identity} stored at 0 degrees.")
                
        elif cmd == 'compare':
            if not self.memory_state_gray:
                self.get_logger().error("Save memory first by sending 'save'.")
                return
            if len(self.memory_state_gray) != len(self.current_cards_gray):
                self.get_logger().error(f"Card count mismatch! Memory has {len(self.memory_state_gray)}, currently see {len(self.current_cards_gray)}.")
                return
                
            self.get_logger().info("[COMPARING STATES]...")
            for i in range(len(self.memory_state_gray)):
                mem_gray = self.memory_state_gray[i]
                cur_gray = self.current_cards_gray[i]
                identity = self.memory_identities[i]
                angle = check_flip_angle(mem_gray, cur_gray)
                
                if 140 < angle < 220:
                    self.get_logger().info(f" -> {identity} (Card {i+1}) CHANGED! (Flipped 180)")
                else:
                    self.get_logger().info(f" -> {identity} (Card {i+1}) did not change.")

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