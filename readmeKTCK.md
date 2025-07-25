Xử Lý Ảnh Với OpenCV – Bộ Bài Tập Python
 Danh sách ảnh sử dụng:
bird.jpg

halong.jpg

pagoda.jpg

 1. Bộ lọc ảnh & chuyển kênh màu (file: basic_filters.py)
Chức năng chính:
Thực hiện các thao tác cơ bản trên từng ảnh:

Median Filter để làm mịn ảnh.

Sobel Filter để phát hiện biên.

Đổi thứ tự kênh màu GRB.

Chuyển ảnh sang không gian màu LAB và tách các kênh L, A, B.

Xử lý thực hiện cho từng ảnh:

Lưu ảnh đã xử lý với tên tương ứng.

Hiển thị trực quan bằng matplotlib.

2. Bộ lọc nâng cao & các thao tác hình thái học (file: advanced_filters.py)
Chức năng:

Cho phép người dùng chọn 1 trong 6 kỹ thuật xử lý ảnh:

B: Gaussian Blur (làm mờ bằng nhân Gauss)

M: Median Blur (làm mờ bằng giá trị trung vị)

F: Bilateral Filter (làm mịn nhưng giữ biên)

E: Canny Edge Detection (tìm biên)

R: Erosion (xói mòn ảnh)

D: Dilation (giãn ảnh)

Chi tiết nổi bật:

Tham số ngẫu nhiên như kernel size, ngưỡng, v.v. được sinh tự động.

Tự động lưu và hiển thị ảnh sau khi xử lý.

Có ghi chú trên ảnh kết quả bằng cv2.putText.

 3. Biến đổi ảnh nâng cao (file: transformations.py)
Mỗi ảnh được xử lý khác nhau theo yêu cầu:

Ảnh 1 – bird.jpg:
 Thêm viền phản chiếu 35px mỗi cạnh.
 Tạo ảnh mới: output_image1_margin.jpg

Ảnh 2 – halong.jpg:
 Xoay ảnh 135 độ quanh tâm rồi lật ngang.
 Tạo ảnh mới: output_image2_rotated_flipped.jpg

Ảnh 3 – pagoda.jpg:
 Phóng to ảnh 5 lần → Làm mờ Gaussian 9x9 → Điều chỉnh độ sáng và tương phản (alpha & beta ngẫu nhiên).
 Tạo ảnh mới: output_image3_final.jpg