# Công Cụ Gán Nhãn Phân Vùng Ảnh (Image Segmentation)

Đây là bộ công cụ hỗ trợ gán nhãn cho bài toán phân vùng ảnh, kết hợp giữa tự động bằng mô hình học máy và chỉnh sửa thủ công.

## 1. Cài đặt

Mở terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:

```bash
pip install opencv-python numpy ultralytics torch torchvision
```

## 2. Cấu hình

Trước khi chạy, chỉnh sửa file `setting/setting_segmentation.txt` với nội dung như sau:

```
folder_input C:/duong/dan/den/thu/muc/anh
forder_output C:/duong/dan/den/thu/muc/luu/nhan
resize 0.5
anpha 1
```

*   `folder_input`: Thư mục chứa ảnh cần gán nhãn.
*   `forder_output`: Thư mục để lưu file nhãn `.txt`.

## 3. Hướng dẫn sử dụng công cụ gán nhãn

File chính để gán nhãn là `auto_labels_segmentation.py` (dùng model YOLO) và `auto_labels_segmentation_ver_2.py` (dùng model Mask R-CNN).

#### Bước 1: Chỉnh đường dẫn model

Mở file `.py` bạn muốn chạy và thay đổi đường dẫn đến file model (`.pt` hoặc `.pth`) cho chính xác.

*   Ví dụ trong `auto_labels_segmentation.py`:
    ```python
    model = YOLO(r"C:\duong\dan\den\model\yolo\best.pt")
    ```

Có thể thay ngưỡng confidence cho phù hợp với yêu cầu.

#### Bước 2: Chạy công cụ

```bash
python auto_labels_segmentation.py
```

#### Bước 3: Quy trình gán nhãn

1.  Tự động tạo mask: Nhấn phím `s` để mô hình học máy tự động tìm các đối tượng. Các vùng chưa gán nhãn sẽ có màu vàng.
2.  Gán nhãn:
    *   Click vào một ô màu (nhãn) ở bảng điều khiển bên phải.
    *   Click vào một vùng mask màu vàng trên ảnh để gán nhãn đó.
3.  Chỉnh sửa thủ công:
    *   Vẽ thêm: Giữ và di chuyển chuột trái.
    *   Xóa bớt: Giữ và di chuyển chuột phải.
    *   Chỉnh kích thước bút/tẩy: Lăn chuột.
4.  Lưu và chuyển ảnh:
    *   Nhấn `d` để lưu nhãn và chuyển sang ảnh tiếp theo.
    *   Nhấn `a` để lưu và quay lại ảnh trước đó.
    *   Nhấn `q` để lưu và thoát.

**Lưu ý**: Bạn phải gán nhãn cho tất cả các vùng (không còn màu vàng) trước khi chuyển ảnh hoặc thoát.

## 4. Các phím tắt chính

`s`: Chạy mô hình tự động phân vùng. 
`d`: Lưu và chuyển sang ảnh tiếp theo. 
`a`: Lưu và quay lại ảnh trước đó. 
`q`: Lưu và thoát. 
`c`: copy toàn bộ mask của ảnh hiện tại.
`v`: dán toàn bộ mask đã copy vào ảnh hiện tại.
`r`: Xóa tất cả nhãn trên ảnh hiện tại. 
`t`: Bật/tắt chế độ di chuyển mask. 

## 5. Các file kiểm tra (Test)
*   `test.py`: Kiểm tra nhanh kết quả của model YOLO trên một ảnh.
*   `test_ver_2.py`: Kiểm tra nhanh kết quả của model Mask R-CNN trên một ảnh.