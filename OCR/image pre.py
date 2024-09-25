import cv2

def preprocess_image(image_path):
    """
    对输入图像进行预处理，包括灰度化、二值化和去噪处理。

    参数:
    image_path (str): 图像的文件路径。

    返回:
    processed_image: 预处理后的图像。
    """
    # 加载图像
    image = cv2.imread(image_path)

    # 检查图像是否成功加载
    if image is None:
        raise ValueError("无法加载图像，请检查路径是否正确。")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 去噪（高斯模糊）
    processed_image = cv2.GaussianBlur(binary, (5, 5), 0)

    return processed_image

# 调用函数，传入图像路径
image_path = '../data_pic/2.jpg'
processed_image = preprocess_image(image_path)

# 显示预处理后的图像（调试用）
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
