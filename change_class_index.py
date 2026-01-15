import os
# Khi bạn muốn gộp các class trong dataset YOLO.
# Ví dụ: dataset ban đầu có 2 class (0 và 1) nhưng bạn muốn chỉ còn 1 class duy nhất (0) để train YOLO.
label_dir = r"C:\20210327\object_recognition\lay_dataset\labels\val"  # <-- thay bằng đường dẫn thật
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Thay class_id = 1 thành 0
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
        
        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("Đã cập nhật toàn bộ file label: class 1 -> class 0.")
