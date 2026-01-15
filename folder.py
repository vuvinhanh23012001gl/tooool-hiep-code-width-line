import os
from pathlib import Path
import json
from lib_main import remove
    
def create_folder(path):
    os.makedirs(path, exist_ok=True)
    print("Đã tạo folder:", path)

def edit_path(input):
    new_path = ""
    for i in list(input):
        if i == str("\\"):
            new_path = new_path + "/"
        if i != str("\\"):
            new_path = new_path + i
    return new_path

def create_folder_parent(name_folder):
    parent = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(parent, name_folder)
    os.makedirs(new_path, exist_ok=True)
    create_folder(new_path)
    return new_path

def create_file_in_folder(folder_path: str, file_name: str):
    """
    Tạo một file mới trong folder_path với tên file_name.
    - Trả về Path nếu file tồn tại hoặc tạo mới.
    - Trả về None nếu lỗi.
    """
    try:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / file_name
        if not file_path.exists():
            file_path.touch()
            print(f"Đã tạo file: {file_path}")
        else:
            print(f"File đã tồn tại: {file_path}")

        return file_path  # TRẢ VỀ PATH CHUẨN

    except Exception as e:
        print(f"❌ Không thể tạo file: {e}")
        return None
        
def read_settings(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, value = line.strip().split(maxsplit=1)
            settings[str(name)] = str(value)
            print(settings[str(name)])
    return settings
def list_all(path):
    return os.listdir(path)

def create_file_in_folder_two(name_file: str, name_folder: str):
            """Tạo ra 1 foder nếu có rồi thì vào đó tạo ra 1 file
             trả về đường dẫn đến file nằm trong folder
            """
            current_dir = os.path.dirname(os.path.abspath(__file__))
            target_dir = os.path.join(current_dir, name_folder)
            os.makedirs(target_dir, exist_ok=True)

            file_path = os.path.join(target_dir, name_file)

            if not os.path.exists(file_path):
                print("📄 File không tồn tại, tạo mới.")
                with open(file_path, "wb") as f:   
                    print("File rỗng")
                    f.write(b"")                   
            return file_path
def write_json_to_file(file_path: str, data: dict, indent: int = 4):
        """
        Ghi dữ liệu dạng JSON vào file.
        - file_path: đường dẫn tới file json
        - data: dict hoặc list cần lưu
        - indent: số khoảng trắng khi format cho dễ đọc
        """
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                print(f"✅ Đã ghi JSON vào: {file_path}")

        except Exception as e:
            print(f"❌ Lỗi khi ghi file JSON: {e}")
            
def read_json_from_file(file_path: str) -> dict:
        """
        Đọc dữ liệu JSON từ file và trả về dạng dict.
        - file_path: đường dẫn tới file JSON
        """
        try:
            # Nếu file chưa tồn tại -> trả về dict rỗng
            if not os.path.exists(file_path):
                print(f"⚠️ File không tồn tại: {file_path}")
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # print(f"✅ Đã đọc JSON từ: {file_path}")
                return data

        except json.JSONDecodeError as e:
            print(f"❌ Lỗi định dạng JSON ({file_path}): {e}")
            return {}
        except Exception as e:
            print(f"❌ Lỗi khi đọc file JSON: {e}")
            return {}
def get_file_path_by_index(folder_path: str, index: int, ext: str = None):
    """
    Lấy đường dẫn file theo index trong folder.
    -1 la phan tu cuoi cung 0 la phan tu dau tien 
    Args:
        folder_path (str): đường dẫn folder.
        index (int): index file (bắt đầu từ 0, có thể âm để đếm từ cuối).
        ext (str, optional): lọc theo đuôi file, ví dụ "pt", "jpg".
    
    Returns:
        str: đường dẫn file, hoặc None nếu không tìm thấy.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print("❌ Folder không tồn tại")
        return None

    files = [f for f in folder.iterdir() if f.is_file()]
    
    # Lọc theo đuôi nếu cần
    if ext:
        ext = ext.lower().lstrip(".")
        files = [f for f in files if f.suffix.lower() == f".{ext}"]

    files.sort()  # sắp xếp theo tên file
    
    # Xử lý index âm
    if index < 0:
        index = len(files) + index

    if index < 0 or index >= len(files):
        print("❌ Index ngoài phạm vi")
        return None

    return str(files[index])

# Function to save polygons to a TXT file
def save_polygons_to_txt(img, file_path, polygons_data_list): # Thêm tham số label
    # polygons_data_list.append({"label": assigned_label, "polygon": current_poly})
    remove.remove_file(file_path)
    if len(polygons_data_list) > 0:
        if len(img.shape) == 3:
            img_height, img_width, _ = img.shape
        else:
            img_height, img_width = img.shape
        data = ""
        with open(file_path, 'w') as f:
            for polygon_data in polygons_data_list: # Đổi tên biến để tránh trùng lặp với tham số polygons
                label = polygon_data["label"]
                list_data = polygon_data["polygon"]
                data = data + label
                for poly in list_data:
                    x = poly[0] / img_width
                    y = poly[1] / img_height
                    # Đảm bảo các giá trị nằm trong khoảng [0, 1]
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    data = data + " " + str(x) + " " + str(y)
                data = data + "\n"
            f.write(f"{data}")

def load_polygons_from_txt(file_path, img_width, img_height):
    global name_label
    all_polygons_data = []  # Thay đổi để lưu trữ danh sách các dictionary
    if not os.path.exists(file_path):
        return [] # Return empty list if file does not exist
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label_loaded = parts[0]
            points_data = parts[1:]

            poly = []
            for i in range(0, len(points_data), 2):
                x = float(points_data[i]) * img_width
                y = float(points_data[i + 1]) * img_height
                poly.append((int(x), int(y)))

            # Tạo một dictionary cho mỗi polygon và thêm vào danh sách
            polygon_entry = {"label": str(label_loaded), "polygon": poly}
            # Dòng dưới đây là nguyên nhân gây lỗi: nó lưu lại tâm của nhãn cũ, khiến chúng bị gán lại sau khi xóa.
            # center = calculate.calculate_center(poly)
            # name_label[str(label_loaded)][1].append(center)
            all_polygons_data.append(polygon_entry)

    return all_polygons_data
def load_current_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                # Lấy dòng đầu tiên là tên ảnh
                return lines[0].strip()
    return None
# Mouse callback function