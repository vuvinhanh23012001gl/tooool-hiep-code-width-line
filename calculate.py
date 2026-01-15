import os
import math
import cv2
import numpy as np


def polygons_hash(polygons):
    normalized = []
    for p in polygons:
        poly = sorted(p["polygon"])
        normalized.append((p["label"], tuple(poly)))

    normalized.sort()  # quan trọng
    return hash(str(normalized))


# Function to rotate a point around a center
def rotate_point(point, center, angle):
    angle_rad = math.radians(angle)
    x, y = point
    cx, cy = center
    x_new = cx + math.cos(angle_rad) * (x - cx) - math.sin(angle_rad) * (y - cy)
    y_new = cy + math.sin(angle_rad) * (x - cx) + math.cos(angle_rad) * (y - cy)
    return int(x_new), int(y_new)


def calculate_center(points):
    if len(points) == 0:
        return None
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return int(center_x), int(center_y)


def polygon_overlap_area(poly1, poly2, h, w):
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.array(poly1, np.int32)], 255)
    cv2.fillPoly(m2, [np.array(poly2, np.int32)], 255)
    return np.sum(cv2.bitwise_and(m1, m2))


def draw_yolo_contours(img, results, color=(0, 255, 0), thickness=2, show_label=True):
    """
    Vẽ đường viền vật thể từ YOLO results lên ảnh gốc.

    Args:
        img (np.ndarray): Ảnh gốc BGR.
        results: Kết quả YOLOv11 (frame đầu tiên).
        color (tuple): Màu vẽ contour (B,G,R).
        thickness (int): Độ dày đường viền.
        show_label (bool): Hiển thị nhãn + confidence nếu model detect.

    Returns:
        np.ndarray: Ảnh đã vẽ contour.
    """
    if results.masks and results.masks.xy:  # model segmentation
        for polygon in results.masks.xy:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    else:  # model detect (không có mask)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            if show_label:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results.names[cls_id]
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def draw_yolo_contours_fill(img, results, color=(0, 255, 0), thickness=2, show_label=True, alpha=0.4):
    """
    Vẽ đường viền vật thể từ YOLO results lên ảnh gốc và đổ đầy màu vào vật thể.

    Args:
        img (np.ndarray): Ảnh gốc BGR.
        results: Kết quả YOLOv11 (frame đầu tiên).
        color (tuple): Màu vẽ contour/đổ fill (B,G,R).
        thickness (int): Độ dày đường viền.
        show_label (bool): Hiển thị nhãn + confidence nếu model detect.
        alpha (float): Độ trong suốt của màu fill (0=trong suốt, 1=đậm).

    Returns:
        np.ndarray: Ảnh đã vẽ contour và fill.
    """
    overlay = img.copy()

    if results.masks and results.masks.xy:  # model segmentation
        for polygon in results.masks.xy:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    else:  # model detect (không có mask)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 để fill
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)  # vẽ viền
            if show_label:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results.names[cls_id]
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Trộn overlay với ảnh gốc để tạo hiệu ứng trong suốt
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img



def load_bbox_state(state_file):
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('show_bbox='):
                        return line.strip().split('=')[1] == 'True'
    except Exception:
        pass
    return False


def create_help_image(text_dict, width=450, height=500):
    """Creates an image with help text."""
    help_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    y_pos = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 25

    # Title
    cv2.putText(help_img, "Hotkeys & Controls", (20, y_pos), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    y_pos += 40

    for key, description in text_dict.items():
        line = f"- {key}: {description}"
        cv2.putText(help_img, line, (20, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y_pos += line_height
        if y_pos > height - 20:
            # In a real scenario, might need multiple columns or a larger image
            break
    return help_img

def get_screen_dimensions():
    """Lấy chiều rộng và chiều cao của màn hình."""
    try:
        # Sử dụng Tkinter để lấy kích thước màn hình một cách an toàn
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        # Giá trị dự phòng nếu Tkinter không khả dụng
        return 1920, 1080
    

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

