# README — convert_to_onnx.ipynb

Notebook này chạy trên **Google Colab** (có GPU) để chuyển checkpoint NAFNet đã train (`.pth`) sang định dạng **ONNX**, phục vụ việc tích hợp model vào ứng dụng mobile (Android/iOS).

## Tại sao phải convert sang ONNX?

- File `.pth` là **checkpoint riêng của PyTorch** — chỉ load được bằng thư viện `torch` trên máy có cài PyTorch (Python). Ứng dụng mobile không chạy được Python/PyTorch trực tiếp trên thiết bị (nặng, không tối ưu cho CPU/NPU điện thoại).
- **ONNX** (Open Neural Network Exchange) là định dạng model trung gian, độc lập framework — mô tả model dưới dạng đồ thị tính toán (graph) tĩnh gồm các phép toán chuẩn (Conv, MatMul, Pad, ...).
- Từ file `.onnx`, có thể chạy trực tiếp bằng **ONNX Runtime Mobile** (Android/iOS) hoặc convert tiếp sang **CoreML** (iOS) / **TFLite**, đều là các runtime được tối ưu để chạy nhanh, nhẹ, tiết kiệm pin trên điện thoại.
- Nói ngắn gọn: `.pth` (dạy/train) → `.onnx` (cầu nối) → runtime mobile (chạy trên điện thoại).

## Vì sao không dùng thẳng model gốc `NAFNetLocal`?

Kiến trúc NAFNet có 2 biến thể:
- `NAFNet` — bản chuẩn, chỉ dùng các phép toán cơ bản (Conv2d, AdaptiveAvgPool2d, PixelShuffle...) → export ONNX ổn định, an toàn.
- `NAFNetLocal` — bản cải tiến dùng khi test (kỹ thuật TLSC), thay `AdaptiveAvgPool2d` bằng phép tính `cumsum` động phức tạp để cho chất lượng tốt hơn ở resolution khác lúc train. Phép tính này khó trace sang ONNX ổn định.

→ Notebook luôn dùng `NAFNet` (không dùng `NAFNetLocal`) khi export, đổi lại là chấp nhận input phải có **kích thước cố định** (`IMG_H` × `IMG_W`) khi export.

## Giải thích từng phần trong notebook

| Mục | Nội dung | Vì sao cần |
|---|---|---|
| **0. Kiểm tra GPU** | `nvidia-smi` | Xác nhận Colab đã cấp GPU (không bắt buộc cho việc export, chỉ giúp load model nhanh hơn) |
| **1. Mount Drive** | `drive.mount(...)` | Notebook chạy trên máy ảo tạm của Colab, mất hết dữ liệu khi hết phiên → phải đọc `.pth` và ghi `.onnx` qua Drive để không mất |
| **2. Clone code + cài deps** | `git clone` repo NAFNet gốc, `pip install` | Cần đúng định nghĩa kiến trúc `NAFNet` (class Python) để dựng lại model rồi mới load được trọng số từ `.pth` — `.pth` chỉ chứa số (tensor), không chứa code kiến trúc |
| **3. Cấu hình (PRESET)** | Chọn đúng `width`, `enc_blk_nums`, `middle_blk_num`, `dec_blk_nums` | Phải khớp *chính xác* với cấu hình lúc train, nếu không việc load checkpoint sẽ báo lỗi `Missing keys`/`Unexpected keys` (đã gặp thực tế: SIDD dùng `[2,2,4,8]`, còn GoPro/REDS dùng `[1,1,1,28]`) |
| **4. Build model + load checkpoint** | `model.load_state_dict(...)` | Gán đúng trọng số đã train vào kiến trúc vừa dựng |
| **5. Export ONNX** | `torch.onnx.export(..., dynamo=False)` | Chuyển model PyTorch sang đồ thị ONNX với input cố định kích thước; `dynamo=False` để dùng exporter kiểu cũ (ổn định hơn với kiến trúc custom, tránh phải cài thêm `onnxscript`) |
| **6. Simplify** | `onnxsim.simplify()` | Gộp/loại bỏ các node thừa trong đồ thị ONNX (hằng số tính sẵn, phép toán trung gian không cần thiết) → file gọn hơn, chạy nhanh hơn trên mobile |
| **7. Verify** | So sánh output ONNX vs PyTorch trên cùng 1 input | Đảm bảo quá trình convert không làm sai lệch kết quả model (chấp nhận sai số rất nhỏ do khác biệt tính toán dấu phẩy động) |
| **8. Test ảnh thật** | Chạy inference bằng ONNX Runtime trên ảnh thật, hiển thị trước/sau | Kiểm tra trực quan chất lượng denoise/deblur *trước khi* tốn công viết code tích hợp vào app — nếu sai từ bước này thì việc build app sẽ vô nghĩa |
| **9. Hoàn tất** | Tổng kết đường dẫn file kết quả + hướng dùng tiếp cho Android/iOS/TFLite | Định hướng bước tiếp theo ngoài phạm vi notebook |

## Các lỗi đã gặp khi chạy thực tế (và cách notebook đã xử lý)

1. **`KeyError: '__version__'`** khi `python setup.py develop` — do clone shallow (`--depth 1`) thiếu git history để sinh version. → Notebook đã bỏ hẳn bước `setup.py develop`, chỉ thêm thư mục repo vào `sys.path` để import trực tiếp.
2. **`Missing keys` / `Unexpected keys`** khi load state dict — do chọn sai `PRESET` (cấu trúc block không khớp checkpoint thật). → Notebook liệt kê sẵn đúng cấu hình cho từng loại checkpoint (SIDD/GoPro/REDS) lấy trực tiếp từ file `options/train/*/NAFNet-*.yml` của repo gốc.
3. **`ModuleNotFoundError: No module named 'onnxscript'`** khi export — do bản `torch` mới trên Colab mặc định dùng exporter mới dựa trên `onnxscript`. → Thêm `dynamo=False` để ép dùng exporter kiểu cũ (không cần `onnxscript`).
4. **Crash khi chạy `onnxsim` qua CLI** (lỗi trong `sympy.factor` khi in bảng thống kê, do model có Tile/ConstantOfShape/Expand tạo tensor hằng số lớn) — việc simplify thực chất đã chạy xong, chỉ crash ở bước hiển thị. → Notebook gọi thẳng Python API `onnxsim.simplify()` thay vì CLI, tránh hẳn đoạn code gây crash.

## Sau khi có file `.onnx`, làm gì tiếp?

- **Android**: dùng thư viện `onnxruntime-android`, load `.onnx` trực tiếp bằng `OrtSession`.
- **iOS**: dùng `onnxruntime-objc`, hoặc convert tiếp sang **CoreML** bằng `coremltools` để tận dụng Neural Engine.
- **TFLite**: convert tiếp bằng `onnx2tf`/`onnx-tf` nếu app dùng nền TensorFlow Lite.
- Nếu model còn nặng: cân nhắc **quantize INT8** (`onnxruntime.quantization.quantize_dynamic`) hoặc dùng bản `width32` thay vì `width64`.

Lưu ý quan trọng: model được export với **input cố định kích thước** (`IMG_H` × `IMG_W`, mặc định 256×256). Khi tích hợp vào app thật, ảnh chụp từ camera cần được resize/crop về đúng kích thước này trước khi đưa vào model, và xử lý lại tỉ lệ khung hình ở phía ứng dụng để tránh ảnh bị méo.
