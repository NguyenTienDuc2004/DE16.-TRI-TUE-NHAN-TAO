import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# --- 1. CẤU HÌNH ĐƯỜNG DẪN (TỰ ĐỘNG TÌM) ---
# Lấy đường dẫn nơi file code này đang nằm
base_dir = os.path.dirname(os.path.abspath(__file__))
# Trỏ vào thư mục raw-img
DATA_DIR = os.path.join(base_dir, 'raw-img')

# Cấu hình huấn luyện
IMG_SIZE = (224, 224) # Kích thước chuẩn của MobileNetV2
BATCH_SIZE = 32       # Số lượng ảnh học một lúc (32 là vừa vặn RAM)
EPOCHS = 10           # Học 10 lần là đủ đẹp để vẽ biểu đồ
NUM_CLASSES = 10      # 10 con vật

# Kiểm tra thư mục
if not os.path.exists(DATA_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục '{DATA_DIR}'")
    print("👉 Hãy đảm bảo thư mục 'raw-img' nằm cùng chỗ với file code này!")
    exit()

print(f"✅ Đã tìm thấy dữ liệu tại: {DATA_DIR}")
print("🚀 Đang khởi tạo bộ nạp dữ liệu (Data Generators)...")

# --- 2. CHUẨN BỊ DỮ LIỆU (QUAN TRỌNG) ---
# Dùng ImageDataGenerator để:
# - Tự động chia 80% học (Train) - 20% kiểm tra (Validation)
# - Chuẩn hóa ảnh theo chuẩn MobileNetV2 (quan trọng!)
# - Tăng cường dữ liệu (Xoay, Lật) để mô hình thông minh hơn

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2, # Chia 20% làm tập kiểm tra
    rotation_range=20,    # Xoay ảnh ngẫu nhiên
    horizontal_flip=True, # Lật ảnh ngang
    zoom_range=0.1        # Phóng to thu nhỏ nhẹ
)

# Bộ tạo dữ liệu Huấn Luyện (Training)
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Bộ tạo dữ liệu Kiểm Tra (Validation)
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# In ra danh sách các lớp để bạn kiểm tra
print("🔍 Các lớp tìm thấy:", list(train_generator.class_indices.keys()))

# --- 3. XÂY DỰNG MÔ HÌNH MOBILENETV2 ---
print("🧠 Đang tải kiến trúc MobileNetV2 (Transfer Learning)...")

# Tải phần "thân" của MobileNetV2 (đã học trên ImageNet)
# include_top=False nghĩa là bỏ phần đầu cũ đi, để mình gắn đầu mới vào
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng phần thân (Freezing)
# Lý do: Để chạy nhanh trên CPU i5 của bạn.
# Nếu không đóng băng, máy sẽ chạy rất lâu.
base_model.trainable = False

# Gắn phần "đầu" mới (Classifier Head) để nhận diện 10 con vật
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x) # Nén đặc trưng
x = Dense(128, activation='relu')(x) # Lớp ẩn
x = Dropout(0.2)(x) # Giảm hiện tượng học vẹt
outputs = Dense(NUM_CLASSES, activation='softmax')(x) # Lớp ra kết quả (10 con)

model = Model(inputs, outputs)

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), # Học chậm để chắc chắn
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. BẮT ĐẦU HUẤN LUYỆN (TRAINING) ---
print(f"🔥 Bắt đầu huấn luyện {EPOCHS} Epochs...")
print("⚠️ Lưu ý: Quá trình này có thể mất 15-20 phút. Đừng tắt máy!")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

# --- 5. LƯU MÔ HÌNH ---
model_name = 'MobileNetV2_DongVat_Final.h5'
model.save(model_name)
print(f"💾 Đã lưu mô hình thành công: {model_name}")

# --- 6. VẼ BIỂU ĐỒ BÁO CÁO ---
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
    plt.title('Độ chính xác (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
    plt.title('Hàm mất mát (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('BieuDo_HuanLuyen_MobileNet.png')
    print("✅ Đã xuất biểu đồ 'BieuDo_HuanLuyen_MobileNet.png'. Mở lên xem ngay!")
    plt.show()

plot_history(history)