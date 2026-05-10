# Hướng Dẫn Chạy Notebook Train

File `train.ipynb` được thiết kế để chạy trên Google Colab với GPU. Notebook thực hiện các bước: mount Google Drive, cài đặt thư viện, clone source code, giải nén dataset, cấu hình đường dẫn, khởi tạo model AdaFace và huấn luyện/evaluate.

## 1. Chuẩn Bị

1. Mở Google Colab và bật GPU:
   - `Runtime` -> `Change runtime type` -> `GPU`.

2. Tải dataset và model/pretrained checkpoint:
   - Dataset: https://drive.google.com/drive/folders/1E4rXfu1p60XOYwgFW31G20OyciwLu5s9?usp=sharing
   - Model: https://drive.google.com/drive/folders/1U-K9FSWov8-qZcDAAM9MNDTaMi0ILL3p?usp=sharing

3. Đặt file dataset trong Google Drive theo đúng đường dẫn mà notebook đang sử dụng, hoặc sửa lại biến đường dẫn trong cell cấu hình.

## 2. Cấu Trúc Đường Dẫn Trong Notebook

Notebook đang cấu hình các biến chính như sau:

```python
TRAIN_DATA_PATH = '/content/dataset/train'
ACCURACY_VAL_PATH = '/content/val_data/data'
CHECKPOINT = '/content/drive/MyDrive/checkpoints/VGG+Asian/ir_50_checkpoint_3.pth'
SAVE_CHECKPOINT_DIR = '/content/drive/MyDrive'
PRE_TRAINED_PATH = '/content/drive/MyDrive/pretrained/adaface_ir50_webface4m.ckpt'
```

Nếu dataset/model nằm ở thư mục khác trên Drive, cần sửa các biến trên cho đúng vị trí file thực tế.

Dataset train cần có dạng thư mục ảnh theo từng identity:

```text
/content/dataset/train/
+-- identity_001/
|   +-- image_1.jpg
|   +-- image_2.jpg
+-- identity_002/
|   +-- image_1.jpg
|   +-- image_2.jpg
+-- ...
```

## 3. Các Bước Chạy

### Bước 1: Mount Google Drive

Chạy cell đầu tiên:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sau khi Colab yêu cầu quyền truy cập, đăng nhập và cấp quyền cho Drive.

### Bước 2: Cài Đặt Thư Viện Và Clone Repo

Chạy cell cài đặt:

```python
!pip install menpo
!pip install fvcore
!apt-get install -y p7zip-full
!git clone https://github.com/im-xiaoming/xiaoying.git
!git clone https://github.com/im-xiaoming/firework.git
```

Nếu repo đã tồn tại trong Colab session, có thể xóa thư mục cũ hoặc restart runtime trước khi chạy lại.

### Bước 3: Giải Nén Dataset

Notebook đang giải nén file:

```python
!7z x /content/drive/MyDrive/datasets/faces_extracted.zip -o/content/data -y
```

Nếu tên file dataset khác, sửa lại đường dẫn file `.zip` và thư mục output. Sau khi giải nén, đảm bảo `TRAIN_DATA_PATH` trỏ đúng tới thư mục train.

### Bước 4: Cấu Hình Training

Kiểm tra và sửa các tham số trong cell cấu hình:

```python
BATCH_SIZE = 256
MODEL_NAME = 'ir_50'  # ir_18, ir_50, vit
LEARNING_RATE = 0.01
EPOCHS = 4

m = 0.4
h = 0.333
s = 64
t_alpha = 0.99
OUTPUT_SIZE = 512
```

Gợi ý:

- Nếu Colab bị tràn VRAM, giảm `BATCH_SIZE` xuống `128`, `64` hoặc `32`.
- Dùng `MODEL_NAME = 'ir_18'` nếu cần train nhanh hơn.
- Dùng `MODEL_NAME = 'ir_50'` hoặc `vit` nếu cần độ chính xác cao hơn và GPU đủ mạnh.

### Bước 5: Khởi Tạo DataLoader, Model, Head Và Optimizer

Chạy các cell import, transform, loader, model, head và optimizer. Notebook sẽ:

- Tạo dataloader cho train/validation.
- Khởi tạo backbone `ir_18`, `ir_50` hoặc `vit`.
- Load pretrained weight nếu `PRE_TRAINED_PATH` khác rỗng.
- Tạo AdaFace head với embedding 512 chiều.
- Dùng optimizer SGD.

### Bước 6: Train Model

Chạy cell:

```python
trainer = Trainer(args, dataloader, model, head, optimizer, criterion)
if CHECKPOINT != '':
    trainer._load_checkpoint(CHECKPOINT)

set_lr(optimizer, 0.001)
trainer.train(save_dir=SAVE_CHECKPOINT_DIR, eval_loader=val_loader, metrics=metrics)
```

Checkpoint mới sẽ được lưu vào `SAVE_CHECKPOINT_DIR`. Nếu muốn train từ đầu, đặt:

```python
CHECKPOINT = ''
```

Nếu muốn fine-tuning từ model pretrained, giữ `PRE_TRAINED_PATH` trỏ tới file `.ckpt` hoặc `.pth`.

### Bước 7: Đánh Giá

Sau khi train xong, chạy cell evaluate:

```python
model.eval()
evaluate.evaluate1(model, val_loader, device, '', '')
```

Nếu đánh giá IJB-B/IJB-C, cần chuẩn bị đúng thư mục `datasets/ijb-testsuite/ijb` và sửa lại đường dẫn trong:

```python
r = evaluate.evaluate2(r'datasets\ijb-testsuite\ijb', model, 'IR', 'IJBB', 512, device=device)
r = evaluate.evaluate2(r'datasets\ijb-testsuite\ijb', model, 'IR', 'IJBC', 512, device=device)
```

## 4. Lỗi Thường Gặp

- `No such file or directory`: kiểm tra lại `TRAIN_DATA_PATH`, `ACCURACY_VAL_PATH`, `CHECKPOINT`, `PRE_TRAINED_PATH`.
- `CUDA out of memory`: giảm `BATCH_SIZE` hoặc dùng backbone nhỏ hơn như `ir_18`.
- `ModuleNotFoundError`: chạy lại cell cài đặt thư viện và clone repo, sau đó restart runtime nếu cần.
- `CLASS_NUM = 0`: dataset train chưa đúng cấu trúc mỗi identity là một folder riêng.

## 5. Kết Quả Đầu Ra

Sau khi train thành công, notebook tạo checkpoint dạng:

```text
ir_18_checkpoint_<epoch>.pth
ir_50_checkpoint_<epoch>.pth
vit_checkpoint_<epoch>.pth
```

File checkpoint này có thể dùng để tiếp tục train, fine-tuning hoặc load lại để evaluate.
