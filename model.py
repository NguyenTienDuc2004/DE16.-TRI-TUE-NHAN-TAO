import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# --- CẤU HÌNH ---
# Tên file mô hình bạn vừa train xong (như trong ảnh bạn gửi)
MODEL_FILE = 'MobileNetV2_DongVat_Final.h5'

# DANH SÁCH NHÃN (QUAN TRỌNG)
# Keras tự động sắp xếp thư mục theo bảng chữ cái ABC.
# Thư mục gốc là tiếng Ý, nên thứ tự 0-9 sẽ là:
# 0: cane (Chó), 1: cavallo (Ngựa), 2: elefante (Voi), 3: farfalla (Bướm),
# 4: gallina (Gà), 5: gatto (Mèo), 6: mucca (Bò), 7: pecora (Cừu),
# 8: ragno (Nhện), 9: scoiattolo (Sóc)

CLASS_NAMES = [
    "CHÓ (Cane)",  # 0
    "NGỰA (Cavallo)",  # 1
    "VOI (Elefante)",  # 2
    "BƯỚM (Farfalla)",  # 3
    "GÀ (Gallina)",  # 4
    "MÈO (Gatto)",  # 5
    "BÒ (Mucca)",  # 6
    "CỪU (Pecora)",  # 7
    "NHỆN (Ragno)",  # 8
    "SÓC (Scoiattolo)"  # 9
]

# --- LOAD MODEL ---
# Tìm file trong cùng thư mục
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, MODEL_FILE)

model = None
if os.path.exists(model_path):
    try:
        print(f"⏳ Đang tải mô hình: {MODEL_FILE}...")
        model = load_model(model_path)
        print("✅ Đã tải xong mô hình!")
    except Exception as e:
        print(f"❌ Lỗi file mô hình: {e}")
else:
    print(f"❌ Không tìm thấy file '{MODEL_FILE}' tại {base_dir}")


class FinalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kiểm Tra Mô Hình Tự Train (MobileNetV2)")
        self.root.geometry("800x850")
        self.root.configure(bg="#ecf0f1")

        # Tiêu đề
        tk.Label(root, text="Test Model: MobileNetV2 Final", font=("Segoe UI", 24, "bold"), bg="#ecf0f1",
                 fg="#2c3e50").pack(pady=20)

        # Trạng thái
        status_text = "SẴN SÀNG" if model else "THIẾU FILE MODEL"
        status_color = "#27ae60" if model else "#c0392b"
        tk.Label(root, text=status_text, font=("Arial", 12, "bold"), fg=status_color, bg="#ecf0f1").pack(pady=5)

        # Nút chọn ảnh
        self.btn_select = tk.Button(root, text="CHỌN ẢNH TEST", command=self.predict,
                                    font=("Arial", 16, "bold"), bg="#8e44ad", fg="white",
                                    padx=40, pady=12, borderwidth=0, cursor="hand2")
        self.btn_select.pack(pady=20)

        # Khung ảnh
        self.image_frame = tk.Frame(root, bg="white", bd=5, relief="flat")
        self.image_frame.pack(pady=10)
        self.canvas = tk.Label(self.image_frame, bg="#bdc3c7", width=500, height=400)
        self.canvas.pack()

        # Kết quả
        self.lbl_result = tk.Label(root, text="Chọn ảnh để kiểm tra...", font=("Segoe UI", 18), fg="#7f8c8d",
                                   bg="#ecf0f1")
        self.lbl_result.pack(pady=30)

    def predict(self):
        if not model:
            messagebox.showerror("Lỗi", "Chưa có file mô hình .h5!")
            return

        path = filedialog.askopenfilename()
        if not path: return

        try:
            # Hiển thị ảnh
            img_disp = Image.open(path).convert('RGB')

            # Resize hiển thị thông minh
            width, height = img_disp.size
            max_dim = 500
            scale = min(max_dim / width, max_dim / height)
            new_w = int(width * scale)
            new_h = int(height * scale)

            img_resized = img_disp.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)

            self.canvas.config(image=photo, width=new_w, height=new_h)
            self.canvas.image = photo

            self.lbl_result.config(text="Đang suy luận...", fg="#e67e22")
            self.root.update()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")
            return

        # --- DỰ ĐOÁN ---
        # 1. Resize đúng chuẩn MobileNetV2 (224x224)
        img = img_disp.resize((224, 224))

        # 2. Chuyển thành mảng và thêm chiều batch
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # 3. Preprocess input (Quan trọng: Đưa về khoảng -1 đến 1)
        x = preprocess_input(x)

        # 4. Model dự đoán
        preds = model.predict(x)

        # 5. Lấy kết quả cao nhất
        idx = np.argmax(preds)  # Vị trí max (0-9)
        score = np.max(preds)  # Điểm số max (0.0 - 1.0)

        # 6. Hiển thị tên lớp tương ứng
        if idx < len(CLASS_NAMES):
            final_name = CLASS_NAMES[idx]

            # Màu sắc dựa trên độ tự tin
            if score > 0.8:
                color = "#27ae60"  # Xanh đậm (Rất chắc)
            elif score > 0.5:
                color = "#f39c12"  # Cam (Tạm ổn)
            else:
                color = "#c0392b"  # Đỏ (Không chắc)

            msg = f"Dự đoán: {final_name}\nĐộ tin cậy: {score * 100:.1f}%"
            self.lbl_result.config(text=msg, fg=color, font=("Segoe UI", 24, "bold"))
        else:
            self.lbl_result.config(text=f"Lỗi Index: {idx}", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = FinalApp(root)
    root.mainloop()