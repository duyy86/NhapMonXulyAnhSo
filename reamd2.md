from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift

# Đường dẫn đến ảnh của bạn
# Đảm bảo ảnh 'world_cup.jpg' nằm cùng thư mục với script này
image_path = 'balloons_noisy.png'

def perform_fft_and_shift(img_array):
    """Thực hiện FFT và dịch chuyển tần số."""
    # Thực hiện FFT
    f_transform = fft2(img_array)
    # Dịch chuyển tần số thấp về trung tâm
    f_shifted = fftshift(f_transform)
    # Lấy biên độ của phổ tần số
    magnitude_spectrum = np.abs(f_shifted)
    return magnitude_spectrum

def apply_butterworth_lowpass_filter(img_array, d0, order):
    """Áp dụng bộ lọc thông thấp Butterworth (BLPF)."""
    M, N = img_array.shape
    H = np.ones((M, N), dtype=float)
    center1 = M / 2
    center2 = N / 2
    
    for i in range(M):
        for j in range(N):
            # Tính khoảng cách Euclidean từ tâm
            r = math.sqrt((i - center1)**2 + (j - center2)**2)
            # Áp dụng công thức BLPF
            H[i, j] = 1 / (1 + (r / d0)**(2 * order))
            
    # Áp dụng bộ lọc trong miền tần số
    f_transform = fft2(img_array)
    f_shifted = fftshift(f_transform)
    
    filtered_spectrum = f_shifted * H
    
    # Dịch chuyển ngược lại và thực hiện IFFT
    f_ishifted = fftshift(filtered_spectrum)
    img_filtered = np.abs(ifft2(f_ishifted))
    
    return img_filtered

def apply_butterworth_highpass_filter(img_array, d0, order):
    """Áp dụng bộ lọc thông cao Butterworth (BHPE)."""
    M, N = img_array.shape
    H = np.ones((M, N), dtype=float)
    center1 = M / 2
    center2 = N / 2
    
    for i in range(M):
        for j in range(N):
            # Tính khoảng cách Euclidean từ tâm
            r = math.sqrt((i - center1)**2 + (j - center2)**2)
            # Áp dụng công thức BHPE (1 - BLPF)
            H[i, j] = 1 - (1 / (1 + (r / d0)**(2 * order)))
            
    # Áp dụng bộ lọc trong miền tần số
    f_transform = fft2(img_array)
    f_shifted = fftshift(f_transform)
    
    filtered_spectrum = f_shifted * H
    
    # Dịch chuyển ngược lại và thực hiện IFFT
    f_ishifted = fftshift(filtered_spectrum)
    img_filtered = np.abs(ifft2(f_ishifted))
    
    return img_filtered

def show_image(image_data, title):
    """Hàm để hiển thị ảnh bằng matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image_data, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Chương trình chính ---
if __name__ == "__main__":
    # Mở ảnh gốc và chuyển đổi sang ảnh xám
    try:
        img_original = Image.open(image_path).convert('L')
        im_array_original = np.asarray(img_original)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy ảnh '{image_path}'. Vui lòng đảm bảo ảnh nằm trong cùng thư mục.")
        exit()

    while True:
        print("\n--- MENU XỬ LÝ ẢNH ---")
        print("F: Biến đổi Fourier và hiển thị phổ tần số")
        print("L: Áp dụng Bộ lọc thông thấp Butterworth (BLPF)")
        print("H: Áp dụng Bộ lọc thông cao Butterworth (BHPE)")
        print("Q: Thoát chương trình")
        
        choice = input("Vui lòng chọn (F/L/H/Q): ").upper()

        if choice == 'F':
            print("Đang thực hiện Biến đổi Fourier (FFT)...")
            magnitude_spectrum = perform_fft_and_shift(im_array_original)
            # Log scale để dễ nhìn hơn phổ tần số
            show_image(np.log(1 + magnitude_spectrum), "Biên độ phổ Fourier (Log scale)")
            print("Đã hiển thị phổ Fourier.")
        elif choice == 'L':
            print("Đang áp dụng Bộ lọc thông thấp Butterworth...")
            # Các tham số cho BLPF
            d0_lowpass = 30.0  # Bán kính cắt tần số
            order_lowpass = 2  # Bậc của bộ lọc
            img_lowpassed = apply_butterworth_lowpass_filter(im_array_original, d0_lowpass, order_lowpass)
            show_image(img_lowpassed, f"Ảnh sau BLPF (d0={d0_lowpass}, order={order_lowpass})")
            print("Đã hiển thị ảnh sau BLPF.")
        elif choice == 'H':
            print("Đang áp dụng Bộ lọc thông cao Butterworth...")
            # Các tham số cho BHPE
            d0_highpass = 30.0 # Bán kính cắt tần số
            order_highpass = 2 # Bậc của bộ lọc
            img_highpassed = apply_butterworth_highpass_filter(im_array_original, d0_highpass, order_highpass)
            show_image(img_highpassed, f"Ảnh sau BHPE (d0={d0_highpass}, order={order_highpass})")
            print("Đã hiển thị ảnh sau BHPE.")
        elif choice == 'Q':
            print("Đang thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

CAU 3:
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



CAU 4:

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
import cv2 

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
            H[i, j] = 1 / (1 + (r / d0_val)**t2)
    
    con = d_fshift * H
    f_ishift = scipy.fftpack.ifftshift(con)
    e = abs(scipy.fftpack.ifft2(f_ishift))

    if np.max(e) > 0:
        e_normalized = (e / np.max(e)) * 255
    else:
        e_normalized = np.zeros_like(e)
    
    return e_normalized.astype(np.uint8)


def _median_filter(image_array_gray, kernel_size=3):
    """
    Thực hiện bộ lọc trung vị.
    kernel_size: kích thước cửa sổ (phải là số lẻ, ví dụ: 3, 5, 7)
    """
    if kernel_size % 2 == 0:
        print("Cảnh báo: Kích thước kernel phải là số lẻ. Sử dụng kernel_size = 3.")
        kernel_size = 3
 
    filtered_image = cv2.medianBlur(image_array_gray, kernel_size)
    return filtered_image


def _min_filter(image_array_gray, kernel_size=3):
    """
    Thực hiện bộ lọc Min (tìm giá trị nhỏ nhất trong cửa sổ).
    kernel_size: kích thước cửa sổ (phải là số lẻ)
    """
    if kernel_size % 2 == 0:
        print("Cảnh báo: Kích thước kernel phải là số lẻ. Sử dụng kernel_size = 3.")
        kernel_size = 3
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    min_filtered_image = cv2.erode(image_array_gray, kernel, iterations=1)
    return min_filtered_image


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


            
