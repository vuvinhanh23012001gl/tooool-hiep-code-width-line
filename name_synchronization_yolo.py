import os

def rename_files(img_folder, txt_folder, new_name):
    # List all image files and corresponding text files
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

    # Ensure corresponding text files exist for each image file
    img_files = [f for f in img_files if os.path.splitext(f)[0] + '.txt' in txt_files]

    # Rename each image file and its corresponding text file
    for i, img_file in enumerate(img_files):
        base_name = f"{new_name}_{i+1}"
        img_src_path = os.path.join(img_folder, img_file)
        txt_src_path = os.path.join(txt_folder, os.path.splitext(img_file)[0] + '.txt')
        
        img_dest_path = os.path.join(img_folder, base_name + os.path.splitext(img_file)[1])
        txt_dest_path = os.path.join(txt_folder, base_name + '.txt')
        
        os.rename(img_src_path, img_dest_path)
        os.rename(txt_src_path, txt_dest_path)
        
        print(f"Renamed {img_file} to {base_name + os.path.splitext(img_file)[1]}")
        print(f"Renamed {os.path.splitext(img_file)[0] + '.txt'} to {base_name + '.txt'}")

# Example usage
img_folder = 'D:/22_labels_segmentation/data_input_output/img_circle_all/circle_2'
txt_folder = 'D:/22_labels_segmentation/data_input_output/img_circle_all/labels_circle_2'
new_name = 'img_new'

rename_files(img_folder, txt_folder, new_name)