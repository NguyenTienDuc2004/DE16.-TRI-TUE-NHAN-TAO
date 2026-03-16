import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# --- CẤU HÌNH ---
MODEL_PATH = 'ResNet50V2_Fixed.h5'  # Tên file model của bạn
TEST_IMAGE = 'con_ga.jpg'  # Thay bằng tên 1 bức ảnh bạn có
IMG_SIZE = (224, 224)

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Đã load model thành công!")
except:
    print(f"Không tìm thấy file {MODEL_PATH}")
    exit()


# ======================================================
# PHẦN 1: VẼ CẤU TRÚC MÔ HÌNH (Dạng tóm tắt)
# ======================================================
def plot_structure():
    # Cách đơn giản nhất để lấy thông số đưa vào báo cáo
    print("\n--- TỔNG QUAN CẤU TRÚC ---")
    model.summary()

    # Nếu bạn đã cài graphviz và pydot, bỏ comment dòng dưới để xuất ảnh sơ đồ
    # tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True)
    # print("Đã lưu sơ đồ cấu trúc vào model_structure.png")


# ======================================================
# PHẦN 2: "MỔ NÃO" - VẼ FEATURE MAPS (Lớp tích chập đầu tiên)
# ======================================================
def visualize_feature_maps():
    print("\nĐang tạo bản đồ đặc trưng...")

    # 1. Tìm lớp Convolution đầu tiên để xem nó "nhìn" thấy gì
    target_layer = None
    for layer in model.layers:
        if 'conv' in layer.name:
            target_layer = layer
            break

    if not target_layer:
        print("Không tìm thấy lớp Convolution nào!")
        return

    # 2. Tạo mô hình trung gian: Input -> Lớp Conv đầu tiên
    feature_model = Model(inputs=model.inputs, outputs=target_layer.output)

    # 3. Xử lý ảnh đầu vào
    try:
        img = image.load_img(TEST_IMAGE, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Chuẩn hóa
    except:
        print(f"Lỗi: Không tìm thấy ảnh {TEST_IMAGE} để mổ xẻ!")
        return

    # 4. Dự đoán (Lấy feature map)
    features = feature_model.predict(x)

    # 5. Vẽ lưới 8x8 (64 filters đầu tiên)
    square = 8
    ix = 1
    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Các đặc trưng thị giác tầng đầu tiên ({target_layer.name})', fontsize=20)

    for _ in range(square):
        for _ in range(square):
            if ix > features.shape[-1]: break
            # Lấy filter thứ ix
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # Vẽ kênh thứ ix-1 (Grayscale)
            plt.imshow(features[0, :, :, ix - 1], cmap='viridis')
            ix += 1

    plt.tight_layout()
    plt.savefig('feature_maps.png')
    print("✅ Đã lưu ảnh phân tích 'feature_maps.png'. Hãy mở lên xem!")
    plt.show()


# ======================================================
# PHẦN 3: GIẢ LẬP BIỂU ĐỒ DỰ ĐOÁN (Bar Chart)
# ======================================================
def plot_prediction_confidence():
    # Đây là biểu đồ cột cho thấy xác suất của từng lớp
    try:
        img = image.load_img(TEST_IMAGE, target_size=IMG_SIZE)
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)[0]

        # Tạo danh sách tên lớp (Giả sử 10 lớp của bạn)
        classes = [f"Class {i}" for i in range(len(preds))]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, preds, color='skyblue')

        # Tô màu đỏ cho cột cao nhất (Dự đoán của model)
        max_idx = np.argmax(preds)
        bars[max_idx].set_color('red')

        plt.title('Phân phối xác suất dự đoán (Confidence Distribution)')
        plt.ylabel('Độ tin cậy (0-1)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)

        # Ghi giá trị lên đầu cột
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval * 100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('prediction_chart.png')
        print("✅ Đã lưu biểu đồ dự đoán 'prediction_chart.png'")
        plt.show()

    except Exception as e:
        print(f"Lỗi vẽ biểu đồ dự đoán: {e}")


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    plot_structure()
    visualize_feature_maps()
    plot_prediction_confidence()