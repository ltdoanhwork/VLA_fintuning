# LIBERO + Isaac-GR00T Integration

Dự án tích hợp dataset LIBERO với model Isaac-GR00T để training và inference.

## 📂 Project Structure

| File | Purpose |
|------|---------|
| **`train.py`** | Script training chính (Config-based) |
| **`evaluate.py`** | Đánh giá model (tính MSE vs Ground Truth) |
| **`visualize.py`** | Vẽ biểu đồ so sánh Predictions vs Ground Truth |
| **`config.py`** | File cấu hình tập trung (Dataset, Model, Training) |
| **`run.sh`** | Launcher script với menu tương tác |
| **`check_env.py`** | Kiểm tra môi trường và dependencies |

## 🚀 Quick Start

### 1. Setup
Kiểm tra môi trường:
```bash
python3 check_env.py
```

### 2. Training
Sử dụng launcher (khuyên dùng):
```bash
./run.sh
```
Hoặc chạy trực tiếp:
```bash
# Quick test
python3 train.py --preset quick_test

# Full training
python3 train.py --preset full_training
```

### 3. Evaluation
Đánh giá model sau khi train (tính MSE):
```bash
python3 evaluate.py --checkpoint output/libero_groot_training/checkpoint-20
```

### 4. Visualization
Xem biểu đồ so sánh action predicted vs ground truth:
```bash
python3 visualize.py --checkpoint output/libero_groot_training/checkpoint-20 --num-samples 5
```

## ⚙️ Configuration

Chỉnh sửa `config.py` để thay đổi:
- **DatasetConfig**: Paths, image size, augmentation.
- **ModelConfig**: Base model, fine-tuning flags.
- **TrainingConfig**: Batch size, learning rate, steps.

## 📊 Results Interpretation

Khi chạy `evaluate.py`, bạn sẽ nhận được MSE (Mean Squared Error):
- **< 0.001**: ⭐⭐⭐ Xuất sắc
- **< 0.01**: ⭐⭐ Rất tốt
- **< 0.1**: ⭐ Tốt (Chấp nhận được)
- **> 0.1**: ⚠️ Cần train thêm

## 🛠️ Troubleshooting

- **Lỗi "Checkpoint not found"**: Kiểm tra đường dẫn checkpoint trong lệnh chạy.
- **Lỗi "CUDA OOM"**: Giảm `batch_size` trong `config.py` hoặc dùng preset `low_memory`.
- **Lỗi Import**: Đảm bảo đã source environment và `Isaac-GR00T` path đúng.

---
Author: ltdoanh
Date: 2025-11-24
