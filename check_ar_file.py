import os

def list_base_names(folder):
    """Lấy danh sách tên file (không có đuôi)."""
    return set(os.path.splitext(f)[0]
               for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f)))

def list_non_matching(folder_a, folder_b):
    base_a = list_base_names(folder_a)
    base_b = list_base_names(folder_b)

    only_in_a = sorted(base_a - base_b)
    only_in_b = sorted(base_b - base_a)

    print("=== FILE KHÔNG TRÙNG ===")

    if only_in_a:
        print("\n❌ Chỉ có trong A:")
        for f in only_in_a:
            print("  -", f)

    if only_in_b:
        print("\n❌ Chỉ có trong B:")
        for f in only_in_b:
            print("  -", f)

    if not only_in_a and not only_in_b:
        print("✔ Không có file lệch — tất cả đều khớp.")

    return only_in_a, only_in_b


# ======== Ví dụ sử dụng ========

folder_images = r"C:\Disk D\Project\Python_Check_Oil\data\img"
folder_labels = r"C:\Disk D\Project\Python_Check_Oil\data\label"

list_non_matching(folder_images, folder_labels)
