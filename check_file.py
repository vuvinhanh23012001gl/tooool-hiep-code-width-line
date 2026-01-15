import os
import shutil

def copy_duplicate_files(folder_a, folder_b, dest_a, dest_b):
    """Tìm file trùng tên (base name) giữa 2 folder và copy chúng sang folder khác"""
    
    # Lấy danh sách file (file name + full path)
    files_a = {os.path.splitext(f)[0]: os.path.join(folder_a, f)
               for f in os.listdir(folder_a)
               if os.path.isfile(os.path.join(folder_a, f))}

    files_b = {os.path.splitext(f)[0]: os.path.join(folder_b, f)
               for f in os.listdir(folder_b)
               if os.path.isfile(os.path.join(folder_b, f))}

    # Tìm base name trùng nhau
    duplicates = set(files_a.keys()) & set(files_b.keys())

    # Tạo thư mục đích nếu chưa có
    os.makedirs(dest_a, exist_ok=True)
    os.makedirs(dest_b, exist_ok=True)

    copied_files = []

    for name in duplicates:
        src_a = files_a[name]
        src_b = files_b[name]

        # Copy file từ folder A sang dest A
        shutil.copy2(src_a, os.path.join(dest_a, os.path.basename(src_a)))
        # Copy file từ folder B sang dest B
        shutil.copy2(src_b, os.path.join(dest_b, os.path.basename(src_b)))

        copied_files.append(name)

    return copied_files
folder_a = r"C:\Disk D\Project\Python_Check_Oil\data\img_and_label_crop_master_detect\crop_master_Detect_800\chia_80_20\val\labels"
folder_b = r"C:\Disk D\Project\Python_Check_Oil\data\img_and_label_crop_master_detect\crop_master_Detect_800\chia_80_20\val\images"
dest_a = r"C:\Disk D\Project\Python_Check_Oil\data\img_and_label_crop_master_detect\crop_master_Detect_800\chia_80_20\val1\labels"
dest_b = r"C:\Disk D\Project\Python_Check_Oil\data\img_and_label_crop_master_detect\crop_master_Detect_800\chia_80_20\val1\images"

result = copy_duplicate_files(folder_a, folder_b, dest_a, dest_b)

print("Các file trùng tên:", result)