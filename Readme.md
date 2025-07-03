###Câu 1 – LangBiang (Otsu Thresholding + dịch ảnh)
Ảnh đầu vào: dalat.jpg

Xử lý:

Cắt vùng (0:400, 0:600) trên ảnh gốc

Dịch vùng ảnh sang phải 100px

Áp dụng Otsu Threshold, nhân thêm hệ số 0.3 để tăng độ nhạy

Lưu ảnh đầu ra thành lang_biang.jpg

Thư viện sử dụng: Pillow, NumPy, matplotlib, skimage.filters

###Câu 2 – Hồ Xuân Hương (Adaptive Thresholding + Xoay)
Ảnh đầu vào: dalat.jpg

Xử lý:

Cắt vùng ROI (400:800, 200:700)

Xoay vùng ROI 45 độ bằng scipy.ndimage.rotate

Áp dụng Adaptive Thresholding (block_size=35, offset=60)

Lưu kết quả thành ho_xuan_huong.jpg

Thư viện sử dụng: Pillow, NumPy, matplotlib, scipy.ndimage, skimage.filters

###Câu 3 – Quảng Trường Lâm Viên (Morphology Closing + mask)
Ảnh đầu vào: quan_truong_lam_vien.jpg

Xử lý:

Chuyển ảnh sang ảnh xám và nhị phân

Tạo mask hình ellipse tại vùng trung tâm

Áp dụng phép binary_closing 3 lần để làm mịn vùng chọn

Áp kết quả vào ảnh gốc để giữ nguyên phần còn lại

Lưu ảnh đầu ra thành quan_truong_lam_vien.jpg

Thư viện sử dụng: OpenCV, NumPy, matplotlib, scipy.ndimage

###Câu 4 – Menu tương tác Biến đổi và Phân ngưỡng
Ảnh đầu vào: quan_truong_lam_vien.jpg

Chức năng:

Cho phép người dùng chọn:

Biến đổi hình học:

a. Rotate (xoay 45 độ)

b. Scale (phóng to 1.5x)

c. Shift (dịch 50px, 30px)

Phân ngưỡng/biến đổi ảnh nhị phân:

d. Adaptive Thresholding

e. Binary Dilation

f. Binary Erosion

g. Otsu Thresholding

Cho phép chọn 1 hoặc 2 thao tác kết hợp

Hiển thị ảnh gốc và kết quả bằng matplotlib

Thư viện sử dụng: OpenCV, NumPy, matplotlib

Tệp cấu trúc

├── dalat.jpg                      # Ảnh gốc đầu vào
├── lang_biang.jpg                 # Kết quả bài 1
├── ho_xuan_huong.jpg             # Kết quả bài 2
├── quan_truong_lam_vien.jpg       # Ảnh gốc và kết quả bài 3 & 4
├── main.ipynb / *.py              # File chứa mã nguồn
├── README.md                      # File mô tả này
