###Bài 1 – Các thao tác xử lý cơ bản với ảnh
Ảnh đầu vào: a.jpg
Xử lý:

Làm mờ trung bình: Áp dụng bộ lọc trung bình (mean filter) với kernel 5x5.

Phát hiện biên: Chuyển sang ảnh xám và áp dụng bộ lọc Laplacian để phát hiện biên.

Đổi màu ngẫu nhiên: Hoán vị ngẫu nhiên các kênh màu RGB.

Tách kênh HSV: Chuyển ảnh sang không gian HSV và lưu riêng từng kênh Hue, Saturation, Value.

Thư viện sử dụng: OpenCV, NumPy
Ảnh đầu ra:

a_mean.jpg

a_edge.jpg

a_random_color.jpg

a_hue.jpg, a_saturation.jpg, a_value.jpg

###Bài 2 – Biến đổi điểm ảnh nâng cao (menu chọn)
Ảnh đầu vào: image1.jpg, image2.jpg, image3.jpg
Chức năng xử lý (dựa trên menu):

I: Image Inverse – Đảo ngược màu ảnh.

G: Gamma Correction – Sửa gamma với giá trị ngẫu nhiên (0.5–2.0).

L: Log Transformation – Biến đổi logarit (hệ số c ngẫu nhiên 1.0–5.0).

H: Histogram Equalization – Cân bằng histogram (kênh Y trong YUV).

C: Contrast Stretching – Giãn tương phản tuyến tính với khoảng ngẫu nhiên.

A: Adaptive Histogram Equalization – CLAHE (theo từng vùng).

Thư viện sử dụng: OpenCV, NumPy
Ảnh đầu ra: Tùy theo phương pháp, lưu dưới tên output_*.jpg

###Bài 3 – Biến đổi hình học & nâng cao ảnh
 3.1 – Resize + Thêm viền ảnh
Ảnh đầu vào: colorful-ripe-tropical-fruits.jpg
Xử lý: Tăng chiều rộng và chiều cao ảnh thêm 30px.
Ảnh đầu ra: output_colorful_resized.jpg

 3.2 – Xoay + Lật ảnh
Ảnh đầu vào: quang-ninh.jpg
Xử lý:

Xoay ảnh 45 độ theo chiều kim đồng hồ

Lật ngang ảnh sau khi xoay
Ảnh đầu ra: output_quangninh_rotated_flipped.jpg

3.3 – Phóng to + Làm mờ
Ảnh đầu vào: pagoda.jpg
Xử lý:

Phóng to ảnh lên 5 lần

Làm mờ Gaussian với kernel 7x7
Ảnh đầu ra: output_pagoda_upscaled_blurred.jpg

 3.4 – Điều chỉnh độ sáng và tương phản
Ảnh đầu vào: pagoda.jpg
Xử lý:

Tăng giảm độ sáng và độ tương phản bằng công thức tuyến tính

Tham số alpha (tương phản) từ 0.5–2.0, beta (độ sáng) từ -50 đến 50
Ảnh đầu ra: output_pagoda_brightness_contrast.jpg