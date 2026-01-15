import os
import shutil

def copy_images_from_subfolders(source_dir, dest_dir):
    """
    Sao chép tất cả các file ảnh từ thư mục nguồn (bao gồm các thư mục con)
    vào một thư mục đích duy nhất.

    Hàm này sẽ tự động xử lý trường hợp tên file bị trùng lặp bằng cách
    thêm một số vào cuối tên file mới.

    :param source_dir: Đường dẫn đến thư mục nguồn chứa ảnh.
    :param dest_dir: Đường dẫn đến thư mục đích để lưu ảnh.
    """
    # Các định dạng file ảnh phổ biến cần tìm
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # 1. Tạo thư mục đích nếu nó chưa tồn tại
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Đã tạo thư mục đích: {dest_dir}")

    copied_count = 0
    # 2. Duyệt qua tất cả các file và thư mục con trong thư mục nguồn
    print(f"Bắt đầu quét thư mục nguồn: {source_dir}")
    for root, _, files in os.walk(source_dir):
        for filename in files:
            # 3. Kiểm tra xem file có phải là file ảnh không (không phân biệt chữ hoa/thường)
            if filename.lower().endswith(image_extensions):
                source_path = os.path.join(root, filename)
                dest_path = os.path.join(dest_dir, filename)

                # 4. Xử lý trường hợp file đã tồn tại ở đích để tránh ghi đè
                if os.path.exists(dest_path):
                    # Tạo tên file mới để không bị trùng
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(dest_dir, new_filename)
                        counter += 1

                # 5. Sao chép file
                try:
                    shutil.copy2(source_path, dest_path)
                    #print(f"Đã sao chép: {source_path} -> {os.path.basename(dest_path)}")
                    copied_count += 1
                except Exception as e:
                    print(f"Lỗi khi sao chép file {source_path}: {e}")

    print(f"\nHoàn tất! Đã sao chép tổng cộng {copied_count} file ảnh vào '{dest_dir}'.")

if __name__ == "__main__":
    # --- THAY ĐỔI CÁC ĐƯỜNG DẪN DƯỚI ĐÂY ---
    
    # Đường dẫn đến thư mục chứa các thư mục con có ảnh của bạn
    thu_muc_nguon = r"C:\20210327\project\Soft\GuiFile-master\GuiFile-master\app\app\training\T0018205" 

    # Đường dẫn đến thư mục bạn muốn lưu tất cả ảnh vào
    thu_muc_dich = r"C:\20210327\labels_segmentation\test\img_3"

    if not os.path.isdir(thu_muc_nguon):
        print(f"Lỗi: Thư mục nguồn '{thu_muc_nguon}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    else:
        copy_images_from_subfolders(thu_muc_nguon, thu_muc_dich)