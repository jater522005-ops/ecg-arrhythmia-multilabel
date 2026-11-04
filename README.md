# ECG Arrhythmia Multilabel Classification

Dự án phân loại đa nhãn ECG 12 chuyển đạo từ bộ dữ liệu [PTB-XL](https://physionet.org/content/ptb-xl/), thực hiện trong khuôn khổ môn Học máy trong Y tế.

Pipeline gồm khám phá dữ liệu, chuẩn hóa ECG theo từng bản ghi và từng chuyển đạo, kết hợp ECG với thông tin tuổi/giới tính, rồi huấn luyện, đánh giá và giải thích mô hình (XAI) trên năm nhóm chẩn đoán PTB-XL.

## Cấu trúc project

```text
ecg-arrhythmia-multilabel/
├── .github/
│   └── agents/
│       └── ptbxl-eda.agent.md       # Hướng dẫn agent hỗ trợ EDA PTB-XL
├── notebooks/
│   ├── 1_eda.ipynb                          # Khám phá metadata và phân bố nhãn
│   ├── 2_preprocessing.ipynb                # Tạo nhãn, kiểm tra split, chuẩn hóa ECG/clinical
│   ├── 3_model_training_and_evaluation.ipynb # Dataset, model, train/validation/test và tối ưu ngưỡng
│   └── 4_xai_and_results_visualization.ipynb # Trực quan hóa kết quả (ROC/PR, Confusion Matrix) và XAI (1D Grad-CAM, Lead Saliency)
├── src/
│   ├── data.py                      # Dataset đa phương thức và kiểm tra leakage
│   ├── engine.py                    # Vòng lặp train/evaluation và metrics
│   └── models.py                    # InceptionTime + BiGRU classifier
├── tests/
│   └── test_training_components.py  # Unit tests cho data/model/training
├── data/                            # Tự tải PTB-XL và sinh dữ liệu tiền xử lý (không commit)
│   └── processed/
├── models/
│   └── checkpoints/                 # Checkpoint sinh khi train (không commit)
├── figures/                         # Hình kết quả cục bộ (không commit)
├── reports/                         # Báo cáo cục bộ (không commit)
├── requirements.txt                 # Thư viện phân tích và tiền xử lý
├── requirements-train.txt           # PyTorch và pytest
├── .gitignore
├── LICENSE
└── README.md
```

Các thư mục `data/`, `models/`, `figures/` và `reports/` được loại khỏi Git. Bộ dữ liệu ECG và checkpoint không được phân phối trong repository; người dùng cần tự tải dữ liệu theo điều khoản PTB-XL và tạo các artifact cần thiết.

## Mô hình và dữ liệu

- **Đầu vào ECG:** 12 chuyển đạo, 500 Hz, 5.000 mẫu cho mỗi bản ghi.
- **Chuẩn hóa ECG:** record-wise, lead-wise z-score; không áp dụng lọc baseline, notch hay wavelet. Giá trị không hữu hạn được thay bằng 0 sau chuẩn hóa.
- **Clinical features:** `age_z` và `sex`. Mean/std của tuổi được fit trên tập train; giá trị tuổi thiếu được điền bằng train mean. `sex` giữ mã nhị phân 0/1.
- **Nhãn đa nhãn:** `NORM`, `MI`, `STTC`, `CD`, `HYP`.
- **Chia tập:** fold 1–8 cho train, fold 9 cho validation, fold 10 cho test; kiểm tra tách biệt theo patient và loại patient thuộc nhóm form/rhythm-only khỏi các split dùng cho modeling.
- **Kiến trúc:** InceptionTime encoder + bidirectional GRU, kết hợp ECG embedding với clinical features.
- **Chọn checkpoint:** validation macro AUPRC. Test set chỉ dùng cho đánh giá cuối cùng.
- **Metrics:** AUROC, AUPRC, Precision, Recall, F1 và Accuracy; hỗ trợ cả ngưỡng mặc định 0,5 lẫn tối ưu ngưỡng theo từng nhãn (Threshold Tuning ưu tiên Recall trên tập validation).
- **XAI & Phân tích kết quả:** 1D Grad-CAM trên tầng InceptionTime cuối, ma trận độ nhạy theo 12 chuyển đạo (Input Gradient Saliency) và đóng góp của nhánh lâm sàng (Multimodal Late Fusion).

## Cài đặt

Các lệnh dưới đây dành cho PowerShell trên Windows; có thể điều chỉnh cách kích hoạt môi trường ảo trên hệ điều hành khác.

```powershell
git clone https://github.com/jater522005-ops/ecg-arrhythmia-multilabel.git
cd ecg-arrhythmia-multilabel

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-train.txt
pip install jupyter
```

`requirements-train.txt` cài phiên bản PyTorch phù hợp mặc định với pip. Nếu huấn luyện bằng GPU NVIDIA, cài PyTorch build tương thích với CUDA và driver trên máy theo [hướng dẫn cài đặt chính thức của PyTorch](https://pytorch.org/get-started/locally/), sau đó kiểm tra:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Chuẩn bị dữ liệu

Tải PTB-XL từ [PhysioNet](https://physionet.org/content/ptb-xl/), chấp thuận điều khoản dữ liệu và đặt cấu trúc cần thiết dưới `data/` ở thư mục gốc dự án. Notebook cần tối thiểu:

```text
data/
├── ptbxl_database.csv
├── scp_statements.csv
└── records500/
    ├── 00000/
    │   └── ..._hr.dat
    └── ...
```

Các tệp tín hiệu WFDB phải đi kèm với header tương ứng. Notebook tìm project root bằng cách dò `data/ptbxl_database.csv`.

## Chạy notebook

Chạy notebook theo thứ tự trong VS Code/Jupyter:

1. [`notebooks/1_eda.ipynb`](notebooks/1_eda.ipynb) — xem schema metadata, phân bố đặc trưng và nhãn.
2. [`notebooks/2_preprocessing.ipynb`](notebooks/2_preprocessing.ipynb) — tạo nhãn, audit split, chuẩn hóa ECG/clinical và ghi artifact vào `data/processed/`.
3. [`notebooks/3_model_training_and_evaluation.ipynb`](notebooks/3_model_training_and_evaluation.ipynb) — tạo Dataset/DataLoader, train, chọn checkpoint theo validation macro AUPRC, đánh giá trên test và tối ưu ngưỡng quyết định theo từng nhãn.
4. [`notebooks/4_xai_and_results_visualization.ipynb`](notebooks/4_xai_and_results_visualization.ipynb) — xuất bảng Classification Report, vẽ đường cong ROC/PR, Multi-label Confusion Matrix, ma trận đồng xuất hiện bệnh lý và giải thích mô hình bằng 1D Grad-CAM & Input Gradient Saliency.

Notebook tiền xử lý tạo lại mảng chuẩn hóa khoảng 5 GB nếu chạy cell materialization; nếu đã có artifact hợp lệ thì không cần chạy lại cell đó. Notebook train yêu cầu các tệp sau:

```text
data/processed/
├── ptbxl_recordwise_leadwise_zscore_500hz.npy
├── ptbxl_recordwise_leadwise_zscore_500hz_index.csv
├── ptbxl_clinical_features_age_zscore_sex_binary.csv
└── ptbxl_clinical_normalization_stats.json
```

Full training trong notebook yêu cầu CUDA; khi đã có sẵn checkpoint `models/checkpoints/inception_bigru_best.pt`, notebook mặc định bỏ qua bước huấn luyện (`FORCE_RETRAIN = False`) để chạy thẳng đánh giá và tối ưu ngưỡng.

## Chạy tests

Từ thư mục gốc dự án:

```powershell
python -m pytest tests -q
```

## Giấy phép

Mã nguồn trong repository được phát hành theo [MIT License](LICENSE). PTB-XL là bộ dữ liệu riêng và tuân theo điều khoản cấp phép của [PhysioNet](https://physionet.org/content/ptb-xl/); giấy phép mã nguồn không thay thế điều khoản của bộ dữ liệu.
