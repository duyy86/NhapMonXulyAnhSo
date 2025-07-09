# Xử lý Ảnh - Bài Tập Nhận Dạng Hình Dạng & Biến Đổi Cơ Bản

## Giới thiệu
Bài thực hành này bao gồm 5 phần, sử dụng các kỹ thuật xử lý ảnh cơ bản như nhị phân hóa, dò cạnh, phát hiện góc, Hough Transform và so khớp đặc trưng giữa hai ảnh. Các thư viện sử dụng bao gồm: Pillow, NumPy, OpenCV, SciPy, scikit-image, matplotlib.

---

## 📌 Câu 2.1 – Gán nhãn đối tượng trong ảnh

- **Ảnh đầu vào**: `geometric.png`
- **Xử lý**:
  - Chuyển ảnh sang ảnh xám
  - Dùng ngưỡng Otsu để phân ngưỡng nhị phân
  - Gán nhãn từng vùng bằng `skimage.morphology.label`
  - Vẽ hình chữ nhật bao quanh từng đối tượng bằng `regionprops`
- **Thư viện sử dụng**: `Pillow`, `NumPy`, `matplotlib`, `skimage`
- **Ảnh đầu ra**: `label_output.png`

---

## 📌 Câu 2.2 – Dò tìm cạnh theo chiều dọc

- **Ảnh đầu vào**: `geometric.png`
- **Xử lý**:
  - Chuyển ảnh xám
  - Dịch ảnh sang phải 1 pixel bằng `scipy.ndimage.shift`
  - Lấy hiệu tuyệt đối giữa ảnh gốc và ảnh dịch để phát hiện biên
- **Thư viện sử dụng**: `Pillow`, `NumPy`, `SciPy`, `matplotlib`

---

## 📌 Câu 2.3 – Dò tìm cạnh với Sobel Filter

- **Ảnh đầu vào**: `geometric.png`
- **Xử lý**:
  - Áp dụng toán tử Sobel theo trục x và y bằng `scipy.ndimage.sobel`
  - Tính tổng độ lớn gradient để tạo bản đồ biên
- **Thư viện sử dụng**: `Pillow`, `NumPy`, `SciPy`, `matplotlib`

---

## 📌 Câu 2.4 – Xác định góc đối tượng (Corner Detection)

- **Ảnh đầu vào**: `geometric.png`
- **Xử lý**:
  - Cài đặt thuật toán Harris Corner Detection thủ công
  - Dùng đạo hàm Sobel, lọc Gaussian và tính công thức Harris Response
- **Thư viện sử dụng**: `Pillow`, `NumPy`, `SciPy`, `matplotlib`

---

## 📌 Câu 2.5 – Dò tìm đường thẳng bằng Hough Transform & So khớp điểm góc giữa 2 ảnh

### 🧭 2.5.1 – Hough Transform thủ công
- **Ảnh đầu vào**: `geometric.png` hoặc ảnh nhị phân chứa đường thẳng
- **Xử lý**:
  - Dò điểm sáng và ánh xạ lên không gian Hough
  - Tính r theo mọi góc θ từ 0–90°
  - Vẽ ảnh không gian Hough
- **Thư viện sử dụng**: `NumPy`, `matplotlib`

### 🧭 2.5.2 – So khớp đặc trưng giữa hai ảnh (Matching Corners)

- **Ảnh đầu vào**: `bird.png` và `geometric.png`
- **Xử lý**:
  - Phát hiện điểm góc bằng Harris
  - Trích xuất đặc trưng từ vùng 5x5 quanh điểm góc
  - Tính khoảng cách Euclidean giữa các đặc trưng
  - Vẽ cặp điểm tương ứng giữa 2 ảnh
- **Thư viện sử dụng**: `scikit-image`, `scipy.spatial.distance`, `matplotlib`, `imageio`

---

## 📁 Cấu trúc thư mục

├── bird.png # Ảnh đầu vào 1
├── geometric.png # Ảnh đầu vào 2
├── label_output.png # Kết quả gán nhãn đối tượng
├── main.ipynb / *.py # File chứa mã nguồn Python
├── README.md # File mô tả này

---

## ✅ Yêu cầu
- Cài đặt các thư viện sau nếu chưa có:
```bash
pip install pillow numpy matplotlib scipy scikit-image imageio
