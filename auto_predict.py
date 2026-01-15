# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from ultralytics import YOLO
import tkinter as tk

# --- Cấu hình ---
# Thay đổi các đường dẫn và giá trị này cho phù hợp với bạn
MODEL_PATH = r"C:\20210327\labels_segmentation\test\yolo11_seg_singleclass_3\weights\best.pt"
IMAGE_FOLDER = r"C:\20210327\labels_segmentation\test\img"
CONFIDENCE_THRESHOLD = 0.7
# Thay đổi tỷ lệ này để cửa sổ hiển thị to hơn hoặc nhỏ hơn
DISPLAY_SCALE_FACTOR = 0.5
# File để lưu trạng thái ảnh đang xem
STATE_FILE = "auto_predict_state.txt"
INFO_PANEL_WIDTH = 400 # Chiều rộng của panel thông tin

# --- Định nghĩa phím tắt ---
HOTKEYS = {
    "d": "Anh tiep theo",
    "a": "Anh quay lai",
    "q": "Thoat chuong trinh"
}

# --- Hàm lấy kích thước màn hình ---
def get_screen_dimensions():
    """Lấy chiều rộng và chiều cao của màn hình."""
    try:
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ chính của tkinter
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        # Giá trị dự phòng nếu tkinter không khả dụng
        return 1920, 1080

# --- Hàm lưu và tải trạng thái ---
def save_state(file_path, image_name):
    """Lưu tên của ảnh đang xem vào file."""
    try:
        with open(file_path, 'w') as f:
            f.write(image_name)
        print(f"Đã lưu trạng thái, lần sau sẽ bắt đầu từ ảnh: {image_name}")
    except Exception as e:
        print(f"Lỗi khi lưu trạng thái: {e}")

def load_state(file_path):
    """Tải tên ảnh từ file trạng thái."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None

def create_info_panel(width, height, hotkeys, mask_count, image_name, stt, total_images):
    """Tạo một cửa sổ panel hiển thị thông tin và phím tắt."""
    panel = np.full((height, width, 3), 240, dtype=np.uint8) # Nền màu xám nhạt
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30
    text_color = (0, 0, 0) # Màu đen
    y_pos = 40

    # Tiêu đề
    cv2.putText(panel, "Thong tin & Phim tat", (20, y_pos), font, 0.8, text_color, 2)
    y_pos += 15
    cv2.line(panel, (20, y_pos), (width - 20, y_pos), text_color, 1)
    y_pos += line_height

    # Thông tin ảnh
    display_name = (image_name[:35] + '...') if len(image_name) > 35 else image_name
    cv2.putText(panel, f"File: {display_name}", (20, y_pos), font, 0.6, text_color, 1)
    y_pos += line_height
    cv2.putText(panel, f"Thu tu: {stt + 1}/{total_images}", (20, y_pos), font, 0.6, text_color, 1)
    y_pos += line_height

    # Số lượng mask
    cv2.putText(panel, f"So luong mask: {mask_count}", (20, y_pos), font, 0.7, (200, 0, 0), 2) # Màu xanh-đỏ cho dễ thấy
    y_pos += 35
    cv2.line(panel, (20, y_pos), (width - 20, y_pos), text_color, 1)
    y_pos += line_height

    # Phím tắt
    for key, desc in hotkeys.items():
        line = f"- Phim '{key}': {desc}"
        cv2.putText(panel, line, (20, y_pos), font, 0.6, text_color, 1)
        y_pos += line_height

    return panel

# --- Tải mô hình YOLO ---
try:
    model = YOLO(MODEL_PATH)
    print("Tải mô hình YOLO thành công.")
except Exception as e:
    print(f"Không thể tải mô hình YOLO: {e}. Vui lòng kiểm tra lại đường dẫn MODEL_PATH.")
    exit()

# --- Lấy danh sách ảnh ---
try:
    # Lọc các tệp có đuôi là ảnh
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    list_name_img = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(image_extensions)])
    
    if not list_name_img:
        print(f"Không tìm thấy ảnh nào trong thư mục: {IMAGE_FOLDER}")
        exit()
    print(f"Tìm thấy {len(list_name_img)} ảnh trong thư mục.")
except FileNotFoundError:
    print(f"Không tìm thấy thư mục ảnh: {IMAGE_FOLDER}. Vui lòng kiểm tra lại đường dẫn IMAGE_FOLDER.")
    exit()

# --- Tạo cửa sổ hiển thị một lần duy nhất ---
WINDOW_NAME = "Ket qua tu dong"
INFO_WINDOW_NAME = "Thong tin"

# Lấy kích thước màn hình một lần để căn giữa cửa sổ
screen_width, screen_height = get_screen_dimensions()

# --- Vòng lặp chính ---
stt = 0  # Chỉ số của ảnh hiện tại

# Tải trạng thái lần cuối
last_image = load_state(STATE_FILE)
if last_image and last_image in list_name_img:
    try:
        stt = list_name_img.index(last_image)
        print(f"Tiếp tục từ ảnh lần trước: {last_image}")
    except ValueError:
        print(f"Không tìm thấy ảnh '{last_image}' trong thư mục, bắt đầu từ đầu.")
        stt = 0

while True:
    # Lấy tên và đường dẫn ảnh
    name_img = list_name_img[stt]
    image_path = os.path.join(IMAGE_FOLDER, name_img)

    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}. Bỏ qua.")
        # Chuyển sang ảnh tiếp theo, xử lý vòng lặp
        stt = (stt + 1) % len(list_name_img)
        continue

    # Tạo một bản sao để vẽ lên, giữ lại ảnh gốc
    img_with_detections = img.copy()

    # --- Dự đoán ---
    print(f"Đang xử lý ảnh: {name_img}...")
    results = model.predict(
        source=image_path,
        imgsz=1024,
        conf=CONFIDENCE_THRESHOLD,
        show=False,  # Tự xử lý việc hiển thị ảnh
        verbose=False
    )

    # --- Vẽ và Đếm ---
    mask_count = 0
    mask_color = [255, 0, 0]  # Màu xanh dương cho mask
    box_color = [0, 255, 0]   # Màu xanh lá cho box
    alpha = 0.4               # Độ trong suốt của mask

    # Kết quả là một danh sách, thường có một phần tử cho một ảnh
    if results and results[0]:
        r = results[0]
        # Kiểm tra xem có mask nào được phát hiện không
        if r.masks is not None:
            mask_count = len(r.masks)

            # Lặp qua từng đối tượng đã phát hiện
            for i in range(mask_count):
                # --- Vẽ Mask ---
                mask_tensor = r.masks.data[i]
                mask_np = mask_tensor.cpu().numpy()
                # Resize mask về kích thước ảnh gốc
                mask_resized = cv2.resize(mask_np, (img_with_detections.shape[1], img_with_detections.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Tạo một lớp phủ màu cho mask
                overlay = img_with_detections.copy()
                overlay[mask_resized > 0.5] = mask_color
                # Trộn lớp phủ với ảnh để tạo độ trong suốt
                img_with_detections = cv2.addWeighted(overlay, alpha, img_with_detections, 1 - alpha, 0)

                # --- Vẽ Bounding Box ---
                if r.boxes is not None and len(r.boxes) > i:
                    box = r.boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), box_color, 2)

    # --- Hiển thị ảnh ---
    h_display, w_display = img_with_detections.shape[:2]
    scaled_h = int(h_display * DISPLAY_SCALE_FACTOR)
    scaled_w = int(w_display * DISPLAY_SCALE_FACTOR)
    display_img = cv2.resize(
        img_with_detections, (scaled_w, scaled_h)
    )

    # --- Tạo và hiển thị panel thông tin ---
    info_panel = create_info_panel(
        width=INFO_PANEL_WIDTH,
        height=scaled_h,  # Chiều cao panel bằng chiều cao ảnh đã resize
        hotkeys=HOTKEYS,
        mask_count=mask_count,
        image_name=name_img,
        stt=stt,
        total_images=len(list_name_img)
    )
    cv2.imshow(INFO_WINDOW_NAME, info_panel)
    cv2.imshow(WINDOW_NAME, display_img)

    # --- Căn giữa 2 cửa sổ trên màn hình ---
    total_width = scaled_w + INFO_PANEL_WIDTH
    start_x = max(0, (screen_width - total_width) // 2)
    start_y = max(0, (screen_height - scaled_h) // 2)

    cv2.moveWindow(INFO_WINDOW_NAME, start_x, start_y)
    cv2.moveWindow(WINDOW_NAME, start_x + INFO_PANEL_WIDTH, start_y)

    # --- Điều khiển bằng bàn phím ---
    # Vòng lặp chờ phím bấm, nhưng vẫn kiểm tra trạng thái cửa sổ liên tục.
    # Điều này đảm bảo chương trình sẽ thoát ngay khi cửa sổ bị đóng.
    while True:
        k = cv2.waitKey(50) & 0xFF # Chờ 50ms rồi kiểm tra lại

        # Điều kiện thoát 1: Người dùng đóng một trong hai cửa sổ bằng nút 'X'.
        # cv2.getWindowProperty trả về < 1 nếu cửa sổ đã bị đóng.
        if (cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 or
            cv2.getWindowProperty(INFO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1):
            k = ord('q') # Giả lập phím 'q' để kích hoạt quy trình thoát
            break

        # Điều kiện thoát 2: Người dùng bấm một phím hợp lệ (a, d, q)
        if k in [ord('a'), ord('d'), ord('q')]:
            break
        # Nếu không có sự kiện gì, vòng lặp tiếp tục chờ.

    # Xử lý hành động sau khi nhận được phím hợp lệ hoặc sự kiện đóng cửa sổ
    if k == ord('q'):
        save_state(STATE_FILE, name_img) # Lưu trạng thái trước khi thoát
        break
    elif k == ord('d'):  # Ảnh tiếp theo
        stt = (stt + 1) % len(list_name_img)
    elif k == ord('a'):  # Ảnh trước đó
        stt = (stt - 1 + len(list_name_img)) % len(list_name_img)

cv2.destroyAllWindows()
print("Đã thoát chương trình.")