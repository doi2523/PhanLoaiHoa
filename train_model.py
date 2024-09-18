import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Đường dẫn tới thư mục dữ liệu
train_data_dir = 'flowers-dataset/train'
test_data_dir = 'flowers-dataset/test'  # Đảm bảo rằng thư mục kiểm tra khác với thư mục huấn luyện

# Khởi tạo ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Dữ liệu huấn luyện
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Dữ liệu kiểm tra
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Chuyển đổi thành tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None, 5])
).repeat()  # Thêm .repeat() để lặp lại dữ liệu

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None, 5])
).repeat()  # Thêm .repeat() để lặp lại dữ liệu

# Xây dựng mô hình
model = Sequential([
    Input(shape=(150, 150, 3)),  # Thay thế input_shape bằng Input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Thêm dropout để giảm overfitting
    Dense(5, activation='softmax')
])

# Biên dịch mô hình
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# Huấn luyện mô hình
model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=test_dataset,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Lưu mô hình
model.save('flowers_model.keras')
