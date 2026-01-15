import os, cv2

# Kiểm tra xem ảnh nào chưa được gán nhãn (.txt) trong thư mục label.
# Hiển thị những ảnh chưa có label để người dùng có thể kiểm tra và bổ sung.
# chuyen dia chi co dau \ sang /
def edit_path(input):
    new_path = ""
    for i in list(input):
        if i == str("\\"):
            new_path = new_path + "/"
        if i != str("\\"):
            new_path = new_path + i
    return new_path
path_phan_mem = edit_path(os.path.dirname(os.path.realpath(__file__)))
if path_phan_mem.split("/")[-1] == "_internal":
    path_phan_mem = path_phan_mem.replace("/_internal","")


def read_settings(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, value = line.strip().split(' ')
            settings[str(name)] = str(value)
            print(settings[str(name)])
    return settings
path_setting = path_phan_mem + "/setting/setting_check_folder.txt"
data_setting = read_settings(path_setting)

# Example usage
image_folder = data_setting["image_folder"]
txt_folder = data_setting["txt_folder"]

list_img = os.listdir(image_folder)
list_txt = os.listdir(txt_folder)

list_name_img = []
for i in range(0,len(list_img)):
    name_img = list_img[i].split(".")[0]
    list_name_img.append(name_img)
list_name_txt = []
for i in range(0,len(list_txt)):
    name_txt = list_txt[i].split(".")[0]
    list_name_txt.append(name_txt)


unlisted_images = [img for img in list_name_img if img not in list_name_txt]
for i in range(0,len(unlisted_images)):
    print(f"Image {unlisted_images[i]} is not labeled yet.")
    img = cv2.imread(os.path.join(image_folder, unlisted_images[i] + '.png'))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    