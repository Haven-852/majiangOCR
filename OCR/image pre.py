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

def detect_edges_and_adjust_rois(processed_image):
    """
    使用边缘检测来动态调整每个区域的大小。

    :param image: 输入图像
    :return: 每个检测到的区域的坐标 (x, y, w, h) 的列表
    """
    # 使用 Canny 边缘检测
    edges = cv2.Canny(processed_image, 30, 100)

    # 显示边缘检测的结果（调试用）
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)

    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"检测到的轮廓数量: {len(contours)}")  # 输出检测到的轮廓数量
    rois = []
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤较小的区域（根据需求调整阈值）
        if w > 50 and h > 50:
            rois.append((x, y, w, h))
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return rois, processed_image


# # 调用函数，传入图像路径
# image_path = '../data_pic/4.jpg'
# processed_image = preprocess_image(image_path)
#
# # 检测边缘并调整区域
# rois, image_with_rois = detect_edges_and_adjust_rois(processed_image)
#
# # 显示检测到的区域
# cv2.imshow('Detected ROIs with Edges', image_with_rois)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 打印检测到的区域坐标
# for idx, roi in enumerate(rois):
#     print(f"ROI {idx + 1}: {roi}")