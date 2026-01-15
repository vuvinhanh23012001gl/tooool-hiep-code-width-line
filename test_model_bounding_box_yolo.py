import cv2
from ultralytics import YOLO

# Load model
# file này để test ouput mô hình đầu ra
# model
model = YOLO(r"C:\Disk D\Project\Python_Check_Oil\training model\tool_labels_segmentation\model\best.pt")
# Đọc ảnh gốc
image_path = r"C:\Disk D\Project\Python_Check_Oil\data\img\img_251208_165902_199.jpg"
img = cv2.imread(image_path)
# Tạo một bản sao để vẽ lên, giữ lại ảnh gốc
img_with_detections = img.copy()
SCALE_SHOW  = 1
# Dự đoán
results = model.predict(
    source=image_path,
    imgsz=1024,
    conf=0.5,
    show=False
)
# Màu cố định cho mask và bounding box
mask_color = [0, 255, 0]  # Màu xanh dương cho mask
box_color = [0, 255, 0]   # Màu xanh lá cho box và text

# Lặp qua các kết quả
for r in results:
    # Kiểm tra xem có mask và bounding box không
    if r.masks is None or r.boxes is None:
        continue
    
    # Lặp qua từng đối tượng đã phát hiện
    for i in range(len(r.boxes)):
        # --- Vẽ Mask ---
        mask_tensor = r.masks.data[i]
        # Chuyển mask sang numpy
        mask = mask_tensor.cpu().numpy()
        # Resize mask về kích thước ảnh gốc
        mask = cv2.resize(mask, (img_with_detections.shape[1], img_with_detections.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Tạo ảnh overlay trong suốt cho mask
        overlay = img_with_detections.copy()
        overlay[mask > 0.5] = mask_color # Áp dụng màu cố định cho vùng mask
        # Pha overlay với ảnh để tạo độ trong suốt
        alpha = 0.5
        img_with_detections = cv2.addWeighted(overlay, alpha, img_with_detections, 1 - alpha, 0)

        # --- Vẽ Bounding Box và Confidence Score ---
        box = r.boxes[i]
        # Lấy tọa độ bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Lấy điểm tin cậy
        conf = float(box.conf[0])

        # Vẽ bounding box
        # cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), box_color, 2)

        # Chuẩn bị và vẽ nhãn (confidence score)
        label = f"Conf: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Vẽ nền cho text
        cv2.rectangle(img_with_detections, (x1, y1 - h - 5), (x1 + w, y1), box_color, -1)
        # Vẽ text
        cv2.putText(img_with_detections, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Resize ảnh hiển thị xuống 50%

display_img = cv2.resize(img_with_detections, (img_with_detections.shape[1] //SCALE_SHOW , img_with_detections.shape[0] // SCALE_SHOW))
# Lưu và hiển thị
cv2.imwrite("ketqua.png", img_with_detections)
cv2.imshow("Kết quả", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
