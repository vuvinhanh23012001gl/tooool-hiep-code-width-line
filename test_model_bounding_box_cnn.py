import torch
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T

# ======== Load model từ file state_dict ========
checkpoint = torch.load(r"C:\Users\anhuv\Desktop\trainging_model\tool_create_hiep\code\labels_segmentation\test\mask_rcnn_state_dict_ver_3.pth", map_location="cpu")
num_classes = checkpoint["num_classes"]

model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ======== Hàm inference 1 ảnh ========
def inference_mask(image_path, model, score_thresh=0.7):
    # Đọc ảnh
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.ToTensor()
    img_tensor = transform(img_rgb)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    
    masks = outputs["masks"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    # Sao chép ảnh để hiển thị kết quả
    result_img = img_bgr.copy()

    # Overlay mask màu xanh lá cây
    for i in range(len(masks)):
        if scores[i] > score_thresh:
            mask = (masks[i, 0] > 0.7).astype(np.uint8)
            mask_3ch = np.stack([mask] * 3, axis=-1)

            # Tạo lớp overlay màu xanh
            color_layer = np.zeros_like(result_img, dtype=np.uint8)
            color_layer[:, :, 1] = 255  # kênh G (xanh lá)

            # Hòa trộn mask
            result_img = np.where(mask_3ch == 1,
                                  cv2.addWeighted(result_img, 0.5, color_layer, 0.5, 0),
                                  result_img)

    return result_img

# ======== Chạy inference ========
image_path = r"C:\Users\anhuv\Desktop\trainging_model\tool_create_hiep\code\img_input\Img_index_train_0.png"
result_img = inference_mask(image_path, model)

# ======== Hiển thị ảnh (1/3 kích thước gốc) ========
h, w = result_img.shape[:2]
display_img = cv2.resize(result_img, (w // 2, h // 2))

cv2.imshow("Mask R-CNN Result", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
