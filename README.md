#####################################333
lab1


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


##############################################



LAB2
from PIL import Image
import math
import scipy
import numpy as np
import imageio.v2 as iio  # Trong một số phiên bản, imageio cần dùng v2
import matplotlib.pyplot as plt

# Mở ảnh và chuyển sang ảnh xám (grayscale)
img = Image.open('bird.png').convert('L')  # Đảm bảo tên ảnh là 'bird.png'

# Chuyển ảnh thành mảng ndarray để xử lý bằng NumPy
im_1 = np.asarray(img)

# Thực hiện phép biến đổi ảnh âm bản (inversion)
im_2 = 255 - im_1

# Chuyển mảng ndarray sau khi biến đổi trở lại thành ảnh
new_img = Image.fromarray(im_2)

# Hiển thị ảnh gốc
img.show()

# Hiển thị ảnh sau khi biến đổi (âm bản)
plt.imshow(new_img, cmap='gray')  # cmap='gray' đảm bảo hiển thị đúng thang độ xám
plt.axis('off')  # Ẩn trục toạ độ để ảnh đẹp hơn
plt.show()
![image](https://github.com/user-attachments/assets/275c21f5-a774-4123-b8f2-2d088a5db901)




from PIL import Image
import math
import scipy
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

# Mở ảnh và chuyển sang ảnh xám (grayscale)
img = Image.open('bird.png').convert('L')  # Đảm bảo tên ảnh là 'bird.png'

# Chuyển ảnh thành mảng NumPy
im_1 = np.asarray(img)

# Khởi tạo giá trị gamma
gamma = 0.5

# Chuyển mảng ảnh từ kiểu int sang float để xử lý chính xác hơn
b1 = im_1.astype(float)

# Tìm giá trị pixel lớn nhất trong ảnh
b2 = np.max(b1)

# Chuẩn hóa giá trị pixel về khoảng [0, 1]
b3 = b1 / b2

# Tính logarit của b3 và nhân với gamma (tính toán phần mũ của hiệu chỉnh gamma)
b2 = np.log(b3) * gamma

# Áp dụng hàm mũ để thực hiện hiệu chỉnh gamma và nhân với 255 để đưa về thang điểm ảnh 8-bit
c = np.exp(b2) * 255.0

# Chuyển kết quả về kiểu số nguyên 8-bit (giá trị pixel từ 0 đến 255)
c1 = c.astype(np.uint8)

# Chuyển mảng NumPy trở lại thành ảnh
d = Image.fromarray(c1)

# Hiển thị ảnh gốc
img.show()

# Hiển thị ảnh sau khi hiệu chỉnh gamma
d.show()

# Hiển thị ảnh bằng matplotlib
plt.imshow(d, cmap='gray')  # Thêm cmap='gray' để hiển thị đúng màu xám
plt.axis('off')  # Tắt trục tọa độ để ảnh rõ hơn
plt.show()
![image](https://github.com/user-attachments/assets/30daa30a-743a-4c53-99b0-b1bd0e9c70d6)



from PIL import Image
import math
import scipy
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

# Đường dẫn đến tệp ảnh
image_path = 'bird.png'

# Thử mở ảnh và chuyển sang ảnh xám (grayscale)
try:
    img = Image.open(image_path).convert('L')  # Đảm bảo ảnh được chuyển về dạng ảnh xám
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

# Chuyển ảnh thành mảng NumPy để xử lý
im_1 = np.asarray(img)

# Chuyển mảng ảnh từ kiểu số nguyên sang số thực để thực hiện tính toán chính xác
b1 = im_1.astype(float)

# Tìm giá trị pixel lớn nhất để dùng trong chuẩn hóa logarit
b2 = np.max(b1)

# Thực hiện biến đổi logarit theo công thức:
# c = (c * log(1 + pixel)) / log(1 + max_pixel), ở đây c = 128 để tăng độ tương phản
c = (128.0 * np.log(1 + b1)) / np.log(1 + b2)

# Chuyển kết quả về kiểu số nguyên 8-bit (0 - 255)
c1 = c.astype(np.uint8)

# Chuyển mảng kết quả về đối tượng ảnh PIL
d = Image.fromarray(c1)

# Hiển thị ảnh gốc
img.show()

# Hiển thị ảnh sau khi biến đổi logarit
d.show()

# Hiển thị ảnh kết quả bằng matplotlib
plt.imshow(d, cmap='gray')  # cmap='gray' giúp hiển thị đúng màu xám
plt.axis('off')             # Tắt hiển thị trục toạ độ để ảnh đẹp hơn
plt.show()
![image](https://github.com/user-attachments/assets/09d91e76-4757-4719-b0a3-6953b904aa84)



from PIL import Image
import math
import scipy
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt

# Mở ảnh và chuyển sang ảnh xám (grayscale)
image_path = 'bird.png'  # Đường dẫn tới ảnh cần xử lý
try:
    img = Image.open(image_path).convert('L')  # Chuyển ảnh sang thang độ xám
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

# Chuyển ảnh sang mảng NumPy 2 chiều (ảnh gốc)
iml = np.asarray(img)

# Làm phẳng mảng ảnh 2 chiều thành 1 chiều
b1 = iml.flatten()

# Tính histogram (tần suất xuất hiện) và các giá trị bin
hist, bins = np.histogram(iml, 256, [0, 255])

# Tính hàm phân phối tích lũy (CDF)
cdf = hist.cumsum()

# Bỏ qua các giá trị CDF bằng 0 để tránh chia cho 0 sau này
cdf_m = np.ma.masked_equal(cdf, 0)

# Thực hiện chuẩn hóa CDF để biến đổi các giá trị pixel
num_cdf_m = (cdf_m - cdf_m.min()) * 255
den_cdf_m = cdf_m.max() - cdf_m.min()
cdf_m = num_cdf_m / den_cdf_m

# Gán lại các giá trị 0 vào vị trí bị mask trong CDF
cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Chuyển thành kiểu uint8 để tạo ảnh

# Dùng CDF để ánh xạ lại giá trị pixel trong ảnh phẳng
im2 = cdf[b1]

# Chuyển mảng 1D trở lại dạng 2D với kích thước ảnh gốc
im3 = np.reshape(im2, iml.shape)

# Tạo ảnh từ mảng kết quả sau khi cân bằng histogram
im4 = Image.fromarray(im3)

# Hiển thị ảnh gốc
img.show()

# Hiển thị ảnh sau khi cân bằng histogram
im4.show()

# Hiển thị ảnh bằng matplotlib
plt.imshow(im4, cmap='gray')  # cmap='gray' giúp hiển thị đúng màu
plt.axis('off')               # Ẩn trục để hiển thị ảnh rõ hơn
plt.show()


![image](https://github.com/user-attachments/assets/f73477ff-f316-47c2-ac9a-1ef458478309)






from PIL import Image
import math  # Không sử dụng trong mã này
import scipy  # Không sử dụng trong mã này
import numpy as np
import imageio.v2 as iio  # Không sử dụng trong mã này
import matplotlib.pyplot as plt

# Mở ảnh và chuyển sang ảnh xám (grayscale)
image_path = 'balloons_noisy.png'  # Đường dẫn tới ảnh cần xử lý
try:
    img = Image.open(image_path).convert('L')  # Chuyển sang ảnh đen trắng
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

# Chuyển ảnh sang mảng NumPy 2 chiều
iml = np.asarray(img)

# Tìm giá trị pixel nhỏ nhất và lớn nhất trong ảnh
a = iml.min()
b = iml.max()
print(f"Giá trị pixel nhỏ nhất (min): {a}, lớn nhất (max): {b}")

# Chuyển mảng ảnh sang kiểu float để xử lý chính xác hơn
c = iml.astype(float)

# Thực hiện biến đổi giãn độ tương phản theo công thức:
# O = (I - min) / (max - min) * 255
# Kiểm tra tránh chia cho 0 nếu ảnh chỉ có 1 mức xám
if (b - a) == 0:
    print("Ảnh có tất cả các pixel cùng một giá trị. Không thể giãn độ tương phản.")
    im2 = np.zeros_like(iml, dtype=np.uint8)  # Tạo ảnh đen (có thể thay bằng iml nếu muốn giữ nguyên)
else:
    im2 = 255 * (c - a) / (b - a)  # Áp dụng công thức giãn độ tương phản
    im2 = im2.astype(np.uint8)    # Chuyển về kiểu uint8 để tạo ảnh PIL

# Chuyển mảng kết quả về ảnh
im3 = Image.fromarray(im2)

# Hiển thị ảnh gốc
img.show()

# Hiển thị ảnh sau khi giãn độ tương phản
im3.show()

# Hiển thị bằng matplotlib
plt.imshow(im3, cmap='gray')  # cmap='gray' đảm bảo hiển thị đúng màu ảnh xám
plt.axis('off')               # Tắt trục để ảnh hiển thị rõ hơn
plt.show()

![image](https://github.com/user-attachments/assets/f34630d9-9d0a-453b-97f0-e124bdeeac78)



from PIL import Image
import math
import scipy.fftpack  # Chỉ cần import phần fftpack từ scipy
import numpy as np
import imageio.v2 as iio  # Không sử dụng trực tiếp trong mã này
import matplotlib.pyplot as plt

# Mở ảnh và chuyển sang ảnh xám (grayscale)
image_path = 'balloons_noisy.png'  # Đường dẫn đến ảnh đầu vào
try:
    img = Image.open(image_path).convert('L')  # Đảm bảo ảnh ở dạng grayscale
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

# Chuyển ảnh thành mảng NumPy để xử lý
iml = np.asarray(img)

# Thực hiện biến đổi Fourier 2D
# Kết quả là một mảng số phức, cần lấy trị tuyệt đối để lấy biên độ
fft_result = abs(scipy.fftpack.fft2(iml))

# Dịch chuyển thành phần tần số thấp về giữa ảnh để dễ quan sát
fft_shifted = scipy.fftpack.fftshift(fft_result)

# Chuyển sang kiểu float để xử lý
fft_shifted = fft_shifted.astype(float)

# Chuẩn hóa ảnh FFT để hiển thị được bằng 8-bit (0-255)
# Nếu không chuẩn hóa, ảnh có thể toàn trắng vì giá trị quá lớn
if np.max(fft_shifted) > 0:
    d_normalized = (fft_shifted / np.max(fft_shifted)) * 255
else:
    d_normalized = np.zeros_like(fft_shifted)  # T_

![image](https://github.com/user-attachments/assets/85385f5a-9ed7-4ced-9417-9bf861b9af23)




from PIL import Image
import math
import scipy.fftpack  # Chỉ import fftpack thay vì toàn bộ scipy
import numpy as np
import imageio.v2 as iio  # Không sử dụng trong đoạn mã này
import matplotlib.pyplot as plt

# Mở ảnh và chuyển thành ảnh xám
image_path = 'balloons_noisy.png'
try:
    img = Image.open(image_path).convert('L')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

# Chuyển ảnh thành mảng numpy
iml = np.asarray(img)

# Thực hiện biến đổi Fourier 2D
fft = scipy.fftpack.fft2(iml)
fft_magnitude = abs(fft)

# Dịch chuyển tần số thấp về giữa ảnh
fft_shifted = scipy.fftpack.fftshift(fft_magnitude)

# Lấy kích thước ảnh (số hàng, số cột)
M, N = fft_shifted.shape
center1 = M / 2
center2 = N / 2

# Khởi tạo bộ lọc thông thấp Butterworth (BLPF)
H = np.ones((M, N))
d_0 = 30.0         # Bán kính cắt tần số
n = 1              # Bậc của bộ lọc Butterworth

for i in range(M):
    for j in range(N):
        r2 = (i - center1)**2 + (j - center2)**2  # Bình phương khoảng cách đến tâm
        r = math.sqrt(r2)
        H[i, j] = 1 / (1 + (r / d_0)**(2 * n))    # Công thức BLPF

# Nhân phổ ảnh với bộ lọc (lọc trong miền tần số)
filtered_fft = fft * H  # Dùng phổ gốc chưa shift để bảo toàn thông tin pha

# Biến đổi Fourier ngược để đưa ảnh về miền không gian
img_reconstructed = abs(scipy.fftpack.ifft2(filtered_fft))

# Chuẩn hóa về khoảng 0–255 để hiển thị
if np.max(img_reconstructed) > 0:
    normalized_img = (img_reconstructed / np.max(img_reconstructed)) * 255
else:
    normalized_img = np.zeros_like(img_reconstructed)

# Chuyển thành ảnh hiển thị được
im_filtered = Image.fromarray(normalized_img.astype(np.uint8))

# Hiển thị ảnh gốc và ảnh sau khi lọc
img.show()
im_filtered.show()

# Hiển thị bằng matplotlib
plt.imshow(im_filtered, cmap='gray')
plt.title("Ảnh sau khi lọc Butterworth thông thấp (BLPF)")
plt.axis('off')
plt.show()


![image](https://github.com/user-attachments/assets/021758c0-854a-4bdf-a700-90a773ae7a67)


cau3#####
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random # Để chọn ngẫu nhiên hoán vị kênh

# --- Hàm thay đổi thứ tự màu RGB ---

def _swap_rgb_channels(image_pil_rgb):
    """
    Hoán đổi ngẫu nhiên thứ tự các kênh màu RGB của ảnh PIL.
    Trả về ảnh PIL đã hoán đổi kênh.
    """
    # Đảm bảo ảnh là chế độ RGB
    if image_pil_rgb.mode != 'RGB':
        image_pil_rgb = image_pil_rgb.convert('RGB')
    
    # Lấy các kênh R, G, B
    r, g, b = image_pil_rgb.split()
    
    # Tạo tất cả các hoán vị có thể của (r, g, b)
    channel_permutations = [
        (r, g, b), # RGB
        (r, b, g), # RBG
        (g, r, b), # GRB
        (g, b, r), # GBR
        (b, r, g), # BRG
        (b, g, r)  # BGR
    ]
    
    # Chọn ngẫu nhiên một hoán vị
    chosen_permutation = random.choice(channel_permutations)
    
    # Hợp nhất các kênh đã hoán đổi
    swapped_img = Image.merge('RGB', chosen_permutation)
    
    return swapped_img

# --- Phần chính của chương trình ---

def main():
    # Cài đặt đường dẫn và tên file ảnh
    image_dir = 'exercise' # Theo yêu cầu từ các câu trước
    image_filename = 'input_image.jpg' # Giả sử tên file ảnh
    image_path = os.path.join(image_dir, image_filename)

    # Đảm bảo thư mục 'exercise' tồn tại
    os.makedirs(image_dir, exist_ok=True)

    # Kiểm tra và tạo ảnh giả nếu không tìm thấy
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy tệp ảnh '{image_filename}' trong thư mục '{image_dir}'.")
        print(f"Đã tạo một ảnh giả lập màu đỏ '{image_filename}' để bạn có thể chạy thử.")
        dummy_img = Image.new('RGB', (200, 200), color = 'red')
        dummy_img.save(image_path)
        # Nếu muốn dừng chương trình nếu không có ảnh:
        # return

    try:
        # Mở ảnh gốc (đảm bảo là ảnh màu RGB)
        img_original_pil = Image.open(image_path)
        if img_original_pil.mode != 'RGB':
            print(f"Cảnh báo: Ảnh '{image_filename}' không ở chế độ RGB. Đã chuyển đổi sang RGB.")
            img_original_pil = img_original_pil.convert('RGB')
        print(f"Đã mở ảnh gốc: {image_path} (Chế độ: {img_original_pil.mode})")

        # Hoán đổi thứ tự màu RGB ngẫu nhiên
        img_swapped_rgb_pil = _swap_rgb_channels(img_original_pil)
        print("Đã hoán đổi ngẫu nhiên thứ tự kênh màu RGB.")
        
        # Lưu ảnh đã biến đổi
        output_filename = "rgb_swapped_image.png"
        output_path = os.path.join(image_dir, output_filename)
        img_swapped_rgb_pil.save(output_path)
        print(f"Ảnh đã hoán đổi kênh được lưu tại: {output_path}")

        # Hiển thị ảnh gốc và ảnh đã hoán đổi kênh
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_original_pil)
        plt.title('Ảnh Gốc (RGB)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_swapped_rgb_pil)
        plt.title('Ảnh Đã Hoán Đổi Kênh RGB')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
    from PIL import Image
import math
import scipy # Giữ nguyên import scipy theo hình ảnh, mặc dù không dùng trực tiếp cho gamma correction
import numpy as np
import imageio.v2 as iio # Giữ nguyên import theo hình ảnh
import matplotlib.pyplot as plt
import os # Thêm import os để xử lý đường dẫn file

#open a grayscale image
image_path = 'pagoda.jpg' # Đã thay đổi tên ảnh
try:
    img = Image.open(image_path).convert('L') # Mở ảnh và chuyển đổi sang thang độ xám ('L')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh '{image_path}'. Vui lòng đảm bảo tệp ảnh tồn tại.")
    exit()

#convert image 1 into an ndarray
im_1 = np.asarray(img)

#init gamma
gamma = 0.5 # Giá trị gamma mặc định như trong ảnh

#convert ndarray from int to float
b1 = im_1.astype(float)

#find maximum value in b1
b2 = np.max(b1)

#b3 is normalized
# Tránh chia cho 0 nếu b2 (max pixel value) là 0 (ảnh hoàn toàn đen)
if b2 == 0:
    print("Cảnh báo: Ảnh hoàn toàn đen. Không thể thực hiện hiệu chỉnh Gamma. Trả về ảnh gốc.")
    c1 = im_1.astype(np.uint8) # Chuyển ảnh gốc về uint8
else:
    b3 = b1 / b2
    # Tránh log(0) nếu có pixel bằng 0 sau khi chuẩn hóa
    b3[b3 == 0] = 1e-10 # Thay thế 0 bằng một số rất nhỏ để tránh log(0)

    #b2 gamma correction exponent is computed
    # Ở đây, biến 'b2' được tái sử dụng để lưu trữ giá trị mũ gamma, hơi khó hiểu nhưng giữ nguyên theo ảnh
    b2_gamma_exponent = np.log(b3) * gamma

    #gamma correction is computed
    c = np.exp(b2_gamma_exponent) * 255.0

    #c1 is converted to type int
    c1 = c.astype(np.uint8) # Đã sửa thành np.uint8 để tránh lỗi TypeError

d = Image.fromarray(c1)

img.show() # Hiển thị ảnh gốc
d.show()   # Hiển thị ảnh sau gamma correction

plt.imshow(d, cmap='gray') # Thêm cmap='gray' để đảm bảo hiển thị đúng màu xám
plt.show()

![image](https://github.com/user-attachments/assets/86239b9e-db3e-4e7c-9756-335f6122fd1f)

![image](https://github.com/user-attachments/assets/46f00414-d3c9-4b80-9014-f45dfc1be7a8)



cau4##################
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random # Để chọn ngẫu nhiên hoán vị kênh

# --- Hàm thay đổi thứ tự màu RGB ---

def _swap_rgb_channels(image_pil_rgb):
    """
    Hoán đổi ngẫu nhiên thứ tự các kênh màu RGB của ảnh PIL.
    Trả về ảnh PIL đã hoán đổi kênh.
    """
    # Đảm bảo ảnh là chế độ RGB
    if image_pil_rgb.mode != 'RGB':
        image_pil_rgb = image_pil_rgb.convert('RGB')
    
    # Lấy các kênh R, G, B
    r, g, b = image_pil_rgb.split()
    
    # Tạo tất cả các hoán vị có thể của (r, g, b)
    channel_permutations = [
        (r, g, b), # RGB
        (r, b, g), # RBG
        (g, r, b), # GRB
        (g, b, r), # GBR
        (b, r, g), # BRG
        (b, g, r)  # BGR
    ]
    
    # Chọn ngẫu nhiên một hoán vị
    chosen_permutation = random.choice(channel_permutations)
    
    # Hợp nhất các kênh đã hoán đổi
    swapped_img = Image.merge('RGB', chosen_permutation)
    
    return swapped_img

# --- Phần chính của chương trình ---

def main():
    # Cài đặt đường dẫn và tên file ảnh
    image_dir = 'exercise' # Theo yêu cầu từ các câu trước
    image_filename = 'input_image.jpg' # Giả sử tên file ảnh
    image_path = os.path.join(image_dir, image_filename)

    # Đảm bảo thư mục 'exercise' tồn tại
    os.makedirs(image_dir, exist_ok=True)

    # Kiểm tra và tạo ảnh giả nếu không tìm thấy
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy tệp ảnh '{image_filename}' trong thư mục '{image_dir}'.")
        print(f"Đã tạo một ảnh giả lập màu đỏ '{image_filename}' để bạn có thể chạy thử.")
        dummy_img = Image.new('RGB', (200, 200), color = 'red')
        dummy_img.save(image_path)
        # Nếu muốn dừng chương trình nếu không có ảnh, bỏ comment dòng 'return' dưới đây:
        # return

    try:
        # Mở ảnh gốc (đảm bảo là ảnh màu RGB)
        img_original_pil = Image.open(image_path)
        if img_original_pil.mode != 'RGB':
            print(f"Cảnh báo: Ảnh '{image_filename}' không ở chế độ RGB. Đã chuyển đổi sang RGB.")
            img_original_pil = img_original_pil.convert('RGB')
        print(f"Đã mở ảnh gốc: {image_path} (Chế độ: {img_original_pil.mode})")

        # Hoán đổi thứ tự màu RGB ngẫu nhiên
        img_swapped_rgb_pil = _swap_rgb_channels(img_original_pil)
        print("Đã hoán đổi ngẫu nhiên thứ tự kênh màu RGB.")
        
        # Lưu ảnh đã biến đổi
        output_filename = "rgb_swapped_image.png"
        output_path = os.path.join(image_dir, output_filename)
        img_swapped_rgb_pil.save(output_path)
        print(f"Ảnh đã hoán đổi kênh được lưu tại: {output_path}")

        # Hiển thị ảnh gốc và ảnh đã hoán đổi kênh
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_original_pil)
        plt.title('Ảnh Gốc (RGB)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_swapped_rgb_pil)
        plt.title('Ảnh Đã Hoán Đổi Kênh RGB')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()

    from PIL import Image
import math
import scipy.fftpack
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
import os
import cv2 # Thêm OpenCV để dễ dàng thực hiện bộ lọc trung vị/min

# --- Bộ lọc thông thấp Butterworth (BLPF) trong miền tần số ---
def _butterworth_lowpass_filter(iml, d0_val, order_val):
    c_fft = scipy.fftpack.fft2(iml.astype(np.float32))
    d_fshift = scipy.fftpack.fftshift(c_fft)

    M, N = d_fshift.shape
    H = np.ones((M, N), dtype=np.float32)
    center1 = M / 2
    center2 = N / 2
    t2 = 2 * order_val

    for i in range(M):
        for j in range(N):
            r = math.sqrt((i - center1)**2 + (j - center2)**2)
            # Công thức bộ lọc thông thấp Butterworth
            H[i, j] = 1 / (1 + (r / d0_val)**t2)
    
    con = d_fshift * H
    f_ishift = scipy.fftpack.ifftshift(con)
    e = abs(scipy.fftpack.ifft2(f_ishift))

    if np.max(e) > 0:
        e_normalized = (e / np.max(e)) * 255
    else:
        e_normalized = np.zeros_like(e)
    
    return e_normalized.astype(np.uint8)

# --- Bộ lọc trung vị (Median Filter) ---
def _median_filter(image_array_gray, kernel_size=3):
    """
    Thực hiện bộ lọc trung vị.
    kernel_size: kích thước cửa sổ (phải là số lẻ, ví dụ: 3, 5, 7)
    """
    if kernel_size % 2 == 0:
        print("Cảnh báo: Kích thước kernel phải là số lẻ. Sử dụng kernel_size = 3.")
        kernel_size = 3
    # cv2.medianBlur hoạt động rất hiệu quả cho bộ lọc trung vị
    filtered_image = cv2.medianBlur(image_array_gray, kernel_size)
    return filtered_image

# --- Bộ lọc Min (Min Filter) ---
def _min_filter(image_array_gray, kernel_size=3):
    """
    Thực hiện bộ lọc Min (tìm giá trị nhỏ nhất trong cửa sổ).
    kernel_size: kích thước cửa sổ (phải là số lẻ)
    """
    if kernel_size % 2 == 0:
        print("Cảnh báo: Kích thước kernel phải là số lẻ. Sử dụng kernel_size = 3.")
        kernel_size = 3
    # Tạo một kernel để thực hiện phép giãn nở (erosion) để tìm min
    # Erosion với kernel vuông sẽ tìm giá trị min trong cửa sổ
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    min_filtered_image = cv2.erode(image_array_gray, kernel, iterations=1)
    return min_filtered_image


# --- Hàm chính điều khiển chương trình ---
def main():
    image_dir = 'exercise'
    image_filename = 'ha-long-bay-in-vietnam.jpg' # Tên ảnh theo yêu cầu trước đó
    image_path = os.path.join(image_dir, image_filename)

    os.makedirs(image_dir, exist_ok=True)

    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy tệp ảnh '{image_filename}' trong thư mục '{image_dir}'.")
        print(f"Đã tạo một ảnh giả lập màu đỏ '{image_filename}' để bạn có thể chạy thử.")
        dummy_img = Image.new('RGB', (200, 200), color = 'red')
        dummy_img.save(image_path)

    try:
        img_pil = Image.open(image_path).convert('L') # Mở ảnh và chuyển sang thang độ xám
        iml = np.asarray(img_pil)
        print(f"Đã mở ảnh gốc: {image_path} (Chế độ: Grayscale)")

        # --- Áp dụng BLPF ---
        d0_val = 30.0 # cut-off radius
        order_val = 1 # order of BLPF
        print(f"Đang áp dụng Bộ lọc thông thấp Butterworth (d0={d0_val}, order={order_val})...")
        filtered_by_blpf_array = _butterworth_lowpass_filter(iml, d0_val, order_val)
        
        # --- Lựa chọn bộ lọc thứ hai: Median Filter hay Min Filter ---
        # Chọn một trong hai dòng dưới đây:
        # filtered_by_second_filter_array = _median_filter(filtered_by_blpf_array, kernel_size=5)
        # print("Đang áp dụng Bộ lọc Trung vị (Median Filter) với kernel 5x5...")
        
        filtered_by_second_filter_array = _min_filter(filtered_by_blpf_array, kernel_size=5)
        print("Đang áp dụng Bộ lọc Min (Min Filter) với kernel 5x5...") #

        final_filtered_pil = Image.fromarray(filtered_by_second_filter_array)

        output_filename = "blpf_and_min_filtered_image.png" # Tên file cho ảnh cuối cùng
        output_path = os.path.join(image_dir, output_filename)
        final_filtered_pil.save(output_path)
        print(f"Ảnh đã lọc được lưu tại: {output_path}")

        # Hiển thị ảnh
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(img_pil, cmap='gray')
        plt.title('Ảnh Gốc (Grayscale)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(filtered_by_blpf_array, cmap='gray')
        plt.title('Ảnh Sau BLPF')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(filtered_by_second_filter_array, cmap='gray')
        plt.title('Ảnh Sau BLPF + Min Filter') # Thay đổi tiêu đề tùy thuộc vào bộ lọc thứ 2
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
![image](https://github.com/user-attachments/assets/a236db0b-9b85-4448-b0ec-3c5a75c021ac)


![image](https://github.com/user-attachments/assets/ff751a75-eed4-4772-9c0d-0a9144fd6ae5)










