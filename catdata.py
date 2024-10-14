from PIL import Image


def split_image(image_path, output_path):
    # 打开原始图片
    image = Image.open(image_path)

    # 获取图片尺寸
    width, height = image.size

    # 计算每个子图的宽度和高度
    sub_width = width // 3
    sub_height = height // 3

    count = 0
    for i in range(3):
        for j in range(3):
            # 计算当前子图的坐标范围
            left = j * sub_width
            upper = i * sub_height
            right = (j + 1) * sub_width
            lower = (i + 1) * sub_height

            # 裁剪当前子图
            sub_image = image.crop((left, upper, right, lower))

            # 将子图保存为独立的图片文件
            sub_image.save(f"{output_path}/sub_image_{count}.png")

            count += 1


# 指定输入图片路径和输出文件夹路径
image_path = "./data/drive66/train/label/21_training.png"
output_path = "./data/lunwen"

# 调用函数进行图片划分
split_image(image_path, output_path)