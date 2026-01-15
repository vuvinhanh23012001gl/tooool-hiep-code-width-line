import os
import shutil
 
def tong_hop_anh(thu_muc_nguon, thu_muc_dich):
    """
    Tổng hợp tất cả các tệp hình ảnh từ thư mục nguồn và các thư mục con của nó
    vào một thư mục đích duy nhất.
 
    Args:
        thu_muc_nguon (str): Đường dẫn đến thư mục chứa các thư mục con có ảnh.
        thu_muc_dich (str): Đường dẫn đến thư mục để lưu tất cả ảnh đã tổng hợp.
    """
    # Các định dạng ảnh phổ biến cần tìm
    dinh_dang_anh = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
 
    # Tạo thư mục đích nếu nó chưa tồn tại
    os.makedirs(thu_muc_dich, exist_ok=True)
    print(f"Đã tạo hoặc xác nhận thư mục đích tại: '{thu_muc_dich}'")
 
    # Duyệt qua tất cả các thư mục và tệp trong thư mục nguồn
    for root, _, files in os.walk(thu_muc_nguon):
        for file in files:
            # Kiểm tra xem tệp có phải là ảnh không
            if file.lower().endswith(dinh_dang_anh):
                duong_dan_nguon = os.path.join(root, file)
                duong_dan_dich = os.path.join(thu_muc_dich, file)
 
                # Xử lý trường hợp tên tệp bị trùng
                if os.path.exists(duong_dan_dich):
                    ten_tep, phan_mo_rong = os.path.splitext(file)
                    counter = 1
                    # Tìm một tên mới chưa tồn tại
                    while True:
                        ten_tep_moi = f"{ten_tep}_{counter}{phan_mo_rong}"
                        duong_dan_dich_moi = os.path.join(thu_muc_dich, ten_tep_moi)
                        if not os.path.exists(duong_dan_dich_moi):
                            duong_dan_dich = duong_dan_dich_moi
                            break
                        counter += 1
               
                # Sao chép tệp ảnh vào thư mục đích
                shutil.copy2(duong_dan_nguon, duong_dan_dich)
                print(f"Đã sao chép: {duong_dan_nguon} -> {duong_dan_dich}")
 
    print("\nHoàn tất quá trình tổng hợp ảnh!")
 
# --- Ví dụ cách sử dụng ---
if __name__ == "__main__":
    folder_nguon = r"C:\tupn\phan_mem\n_kiem_tra_hinh_anh_dau\training\training\SanPham1"  # THAY ĐỔI ĐƯỜNG DẪN NÀY
    folder_dich = r"C:\tupn\phan_mem\n_kiem_tra_hinh_anh_dau\training\data\sp_1"          # THAY ĐỔI ĐƯỜNG DẪN NÀY
    tong_hop_anh(folder_nguon, folder_dich)