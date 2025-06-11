from PIL import Image  # Nhập thư viện Pillow để xử lý ảnh
import numpy as np     # Nhập thư viện NumPy để xử lý mảng (hiện chưa dùng đến trong đoạn mã này)

img = Image.open('bird.png')  # Mở tệp hình ảnh có tên là 'bird.png'
img.show()                    # Hiển thị ảnh ra bằng trình xem mặc định



import numpy as np                    # Nhập thư viện NumPy để xử lý mảng (hữu ích cho dữ liệu ảnh)
import imageio.v2 as iio             # Nhập thư viện imageio (phiên bản 2) để đọc ảnh
import matplotlib.pylab as plt       # Nhập Matplotlib để hiển thị ảnh

data = iio.imread('bird.png')        # Đọc ảnh 'bird.png' và lưu vào biến 'data' dưới dạng mảng NumPy
plt.imshow(data)                     # Hiển thị ảnh bằng Matplotlib
plt.show()                           # Hiện cửa sổ đồ họa chứa ảnh
![image](https://github.com/user-attachments/assets/04517e00-2883-45a6-a6b3-3afd50ebe31b)




import numpy as np                    # Thư viện xử lý mảng số
import imageio.v2 as iio             # Thư viện đọc ảnh
import matplotlib.pylab as plt       # Thư viện vẽ ảnh

data = iio.imread('bird.png', mode='F')  # Đọc ảnh dưới dạng float grayscale (mức xám 32-bit float)
plt.imshow(data)                         # Hiển thị ảnh
plt.show()                               # Hiện cửa sổ ảnh
![image](https://github.com/user-attachments/assets/45e81465-d20e-47aa-9acd-900b404a1072)




import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
import os

# Đảm bảo thư mục 'lab1_img' tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám và chuyển sang kiểu uint8 (8-bit)
data = iio.imread('bird.png', mode='F').astype(np.uint8)

# Áp dụng mặt nạ bit: giữ lại 4 bit cao (bitmask 0xF0)
cl = data & 0xF0

# Lưu ảnh kết quả vào thư mục lab1_img
iio.imsave('lab1_img/birdf0.png', cl)

# Đọc lại ảnh đã lưu
tmp = iio.imread('lab1_img/birdf0.png')

# Hiển thị ảnh với màu xám
plt.imshow(tmp, cmap='gray')
plt.axis('off')  # Ẩn trục tọa độ
plt.show()
![image](https://github.com/user-attachments/assets/3ffe49ed-e685-4c6b-9ae8-7558050e94bd)





import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt

# Đọc ảnh RGB
data = iio.imread('bird.png')  # Kích thước: (cao, rộng, 3)

# Tạo ảnh mới bằng cách cộng kênh xanh lá (G) và xanh dương (B)
bdata = data[:, :, 1] + data[:, :, 2]  # data[:,:,1] là G, data[:,:,2] là B

# Hiển thị ảnh kết quả
plt.imshow(bdata)
plt.show()
![image](https://github.com/user-attachments/assets/29e6ed9c-9f94-4f6d-a0c7-e39cc9d50b8a)




import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
import colorsys

# Chuyển đổi màu đỏ (255, 0, 0) từ RGB sang HSV
a = colorsys.rgb_to_hsv(255, 0, 0)
print(a)

# Chuyển màu đỏ rất tối (1, 0, 0) sang HSV
b = colorsys.rgb_to_hsv(1, 0, 0)
print(b)

# Chuyển màu xanh lá (0, 255, 0) sang HSV
c = colorsys.rgb_to_hsv(0, 255, 0)
print(c)

# Chuyển từ HSV về RGB với hue=1, saturation=1, value=255
d = colorsys.hsv_to_rgb(1, 1, 255)
print(d)
![image](https://github.com/user-attachments/assets/579bb56c-c13b-43e9-a01f-829d2e98f72b)



import colorsys

# Chuyển màu đỏ (255, 0, 0) từ RGB sang HSV
a = colorsys.rgb_to_hsv(255, 0, 0)
print(a)

# Chuyển màu đỏ rất tối (1, 0, 0) từ RGB sang HSV
b = colorsys.rgb_to_hsv(1, 0, 0)
print(b)

# Chuyển màu xanh lá (0, 255, 0) từ RGB sang HSV
c = colorsys.rgb_to_hsv(0, 255, 0)
print(c)

# Chuyển đổi ngược lại từ HSV (h=1, s=1, v=255) sang RGB
d = colorsys.hsv_to_rgb(1, 1, 255)
print(d)
![image](https://github.com/user-attachments/assets/bbdb5910-db3d-498b-b5e5-5b7035983b0d)





import numpy as np
import imageio.v2 as iio
import matplotlib.pylab as plt
import colorsys

# Đọc ảnh RGB
rgb = iio.imread('bird.png')

# Vectorize hàm chuyển đổi rgb_to_hsv của colorsys (nhận và trả về từng pixel)
rgb2hsv = np.vectorize(colorsys.rgb_to_hsv)

# Chuyển từng kênh R,G,B sang HSV
h, s, v = rgb2hsv(rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])

# Thay đổi giá trị Hue (h), nhân với chính nó để biến đổi màu sắc
h = h * h

# Vectorize hàm chuyển đổi hsv_to_rgb của colorsys
hsv2rgb = np.vectorize(colorsys.hsv_to_rgb)

# Chuyển lại HSV thành RGB
rgb2 = hsv2rgb(h, s, v)

# Chuyển kết quả sang định dạng mảng NumPy với đúng shape (chiều cao, rộng, kênh)
rgb2 = np.array(rgb2).transpose((1, 2, 0))

# Hiển thị ảnh kết quả
plt.imshow(rgb2)
plt.axis('off')
plt.show()
![image](https://github.com/user-attachments/assets/365a1a9a-4a01-4e95-8434-891ee7798dcb)





import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh ở chế độ float (mức xám)
a = iio.imread('bird.png', mode='F')

# Tạo kernel trung bình 5x5
k = np.ones((5, 5)) / 25

# Áp dụng lọc trung bình (tích chập)
b = sn.convolve(a, k).astype(np.uint8)

# Lưu ảnh kết quả
iio.imwrite('lab1_img/bird_mean_filter.png', b)

# In dữ liệu ảnh đã lọc
print(b)

# Hiển thị ảnh kết quả
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()

![image](https://github.com/user-attachments/assets/5ee24179-ddd4-46b7-9165-e7e9a75d987e)





import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám, chuyển sang uint8
a = iio.imread('bird.png', mode="F").astype(np.uint8)

# Áp dụng bộ lọc trung vị với cửa sổ 5x5
b = sn.median_filter(a, size=5, mode='reflect')

# Lưu ảnh đã lọc
iio.imwrite('lab1_img/bird_median_filter.png', b)

# In dữ liệu ảnh sau lọc
print(b)

# Hiển thị ảnh kết quả
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()

![image](https://github.com/user-attachments/assets/a74e85cf-494a-43c1-9944-435a098cee44)





import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám, chuyển sang uint8
a = iio.imread('bird.png', mode='F').astype(np.uint8)

# Áp dụng bộ lọc cực đại với cửa sổ 5x5
b = sn.maximum_filter(a, size=5, mode='reflect')

# Lưu ảnh đã lọc
iio.imwrite('lab1_img/bird_max_filter.png', b)

# In dữ liệu ảnh sau lọc
print(b)

# Hiển thị ảnh kết quả
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()
![image](https://github.com/user-attachments/assets/6e52a5ba-cb44-4189-be94-ebdda31c5cd6)





import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám, chuyển sang uint8
a = iio.imread('bird.png', mode="F").astype(np.uint8)

# Áp dụng bộ lọc cực tiểu với cửa sổ 5x5
b = sn.minimum_filter(a, size=5, mode='reflect')

# Lưu ảnh đã lọc
iio.imwrite('lab1_img/bird_min_filter.png', b)

# In dữ liệu ảnh sau lọc
print(b)

# Hiển thị ảnh kết quả
plt.imshow(b, cmap='gray')
plt.axis('off')
plt.show()
![image](https://github.com/user-attachments/assets/a317b035-795d-48c5-b21f-47771ec27f59)







import numpy as np
import imageio.v2 as iio
from skimage import filters
import matplotlib.pylab as plt
import os

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám, kiểu float
a = iio.imread('bird.png', mode='F')

# Áp dụng bộ lọc Sobel để phát hiện biên
b = filters.sobel(a)

# Chuẩn hóa ảnh kết quả và chuyển sang uint8 (0-255)
b_uint8 = (b / b.max() * 255).astype(np.uint8)

# Lưu ảnh đã xử lý
iio.imwrite('lab1_img/bird_sobel_filter_edge_detection.png', b_uint8)

# Hiển thị ảnh biên
plt.imshow(b_uint8, cmap='gray')
plt.axis('off')
plt.show()

xxx




import numpy as np
import imageio.v2 as iio
from skimage import filters
import matplotlib.pylab as plt
import os

os.makedirs('lab1_img', exist_ok=True)

a = iio.imread('bird.png', mode="F")

# Áp dụng bộ lọc Prewitt
b = filters.prewitt(a)

# Chuẩn hóa kết quả về 0-255 rồi chuyển sang uint8
b_uint8 = (b / b.max() * 255).astype(np.uint8)

# Lưu ảnh kết quả
iio.imwrite('lab1_img/bird_prewitt_filter_edge_detection.png', b_uint8)

# Hiển thị ảnh với màu xám
plt.imshow(b_uint8, cmap='gray')
plt.axis('off')
plt.show()


![image](https://github.com/user-attachments/assets/81dc85e8-e818-421e-b6b8-81dbd7adf06f)







import numpy as np
import imageio.v2 as iio
from skimage import feature
import matplotlib.pylab as plt
import os

os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh ở chế độ mức xám float
a = iio.imread('bird.png', mode="F")

# Phát hiện biên bằng Canny, sigma điều chỉnh độ mượt
b = feature.canny(a, sigma=3)

# Chuyển ảnh nhị phân True/False sang 0-255 uint8 để lưu ảnh rõ hơn
b_uint8 = (b * 255).astype(np.uint8)

# Lưu ảnh kết quả
iio.imwrite('lab1_img/bird_canny_filter_edge_detection.png', b_uint8)

# Hiển thị ảnh với màu xám
plt.imshow(b_uint8, cmap='gray')
plt.axis('off')
plt.show()

![image](https://github.com/user-attachments/assets/7161bd13-6c92-430c-8b21-c2a3b9d58a2a)




import numpy as np
import imageio.v2 as iio
import scipy.ndimage as sn
import matplotlib.pylab as plt
import os

os.makedirs('lab1_img', exist_ok=True)

# Đọc ảnh mức xám float
a = iio.imread('bird.png', mode="F")

# Áp dụng bộ lọc Laplace
b = sn.laplace(a, mode='reflect')

# Chuẩn hóa về 0-255 (dùng giá trị tuyệt đối để tránh số âm)
b_norm = np.abs(b)
b_norm = (b_norm / b_norm.max()) * 255
b_uint8 = b_norm.astype(np.uint8)

# Lưu ảnh kết quả
iio.imwrite('lab1_img/bird_laplace_filter_edge_detection.png', b_uint8)

# Hiển thị ảnh biên với màu xám
plt.imshow(b_uint8, cmap='gray')
plt.axis('off')
plt.show()


![image](https://github.com/user-attachments/assets/4213cc83-6ce7-4825-ae1e-22cd4ba95b7f)



















