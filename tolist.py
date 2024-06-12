import os

def get_file_names_in_folder(folder_path):
    # 获取文件夹中所有文件名
    file_names = os.listdir(folder_path)
    return file_names

def save_names_to_txt(file_names, output_file):
    with open(output_file, 'w') as file:
        for name in file_names:
            file.write(name + '\n')

# 指定文件夹路径
folder_path = "/home/a16/zyx/TransUNet-main/prompt/image/clintox"
output_file = "/home/a16/zyx/TransUNet-main/lists/prompt/clintoxlist.txt"

# 获取文件夹中所有文件名
file_names = get_file_names_in_folder(folder_path)

# 将文件名保存到txt文件中
save_names_to_txt(file_names, output_file)

print("File names have been saved to", output_file)