import os
import shutil
import random

def split_data(img_dir, label_dir, output_dir, train_ratio=0.8):
    """
    Splits image and label data into training and validation sets.

    Args:
        img_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing corresponding label files.
        output_dir (str): Path to the directory where train and val folders will be created.
        train_ratio (float): Ratio of data to be used for training (default: 0.8).
    """

    # Xóa thư mục output cũ nếu nó tồn tại
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Đã xóa thư mục output cũ: {output_dir}")

    # Create output directories if they don't exist
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get list of image files
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # Shuffle the files randomly
    #random.shuffle(img_files)

    # Split the files into train and validation sets
    train_size = int(len(img_files) * train_ratio)
    train_files = img_files[:train_size]
    val_files = img_files[train_size:]

    # Copy files to respective directories
    def copy_files(files, img_source_dir, label_source_dir, img_target_dir, label_target_dir):
        for img_file in files:
            # Extract the base name of the file
            file_name, file_ext = os.path.splitext(img_file)
            
            # Construct the corresponding label file name
            label_file = file_name + '.txt'  # Assuming .txt extension for labels
            
            # Construct the full paths for source and destination
            img_source_path = os.path.join(img_source_dir, img_file)
            label_source_path = os.path.join(label_source_dir, label_file)
            img_target_path = os.path.join(img_target_dir, img_file)
            label_target_path = os.path.join(label_target_dir, label_file)
            
            # Check if the label file exists before attempting to copy
            if os.path.exists(label_source_path):
                shutil.copy2(img_source_path, img_target_path)  # copy2 preserves metadata
                shutil.copy2(label_source_path, label_target_path)
                #print(f"Copied {img_file} and {label_file} to {img_target_dir} and {label_target_dir}")
            else:
                print(f"Warning: Label file {label_file} not found for image {img_file}")

    copy_files(train_files, img_dir, label_dir, train_img_dir, train_label_dir)
    copy_files(val_files, img_dir, label_dir, val_img_dir, val_label_dir)

# Ví dụ sử dụng:
img_dir = r"C:\Disk D\Project\Python_Check_Oil\data\img"
label_dir = r"C:\Disk D\Project\Python_Check_Oil\data\label"
output_dir = r"C:\Disk D\Project\Python_Check_Oil\data\output"
# dataset_5/
# ├─ images/
# │  ├─ train/      # 80% ảnh được copy vào đây
# │  └─ val/        # 20% ảnh còn lại vào đây
# └─ labels/
#    ├─ train/      # file .txt tương ứng với ảnh train
#    └─ val/        # file .txt tương ứng với ảnh val
split_data(img_dir, label_dir, output_dir)
