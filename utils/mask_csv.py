import os
import csv

# 配置参数
SOURCE_DIR = "/mnt/workspace/yangnq/refer/DiffSynth-Studio_sr/data_5000"
OUTPUT_FILE = "/mnt/workspace/lihaoran/data_process/test_data/data_5000_all.csv"
USE_ABSOLUTE_PATH = True  # True: 绝对路径, False: 相对路径

def generate_csv():
    """遍历目录生成包含MP4文件路径的CSV文件"""
    count = 0
    
    # 使用utf-8编码，newline=''避免Windows下出现空行
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["video_path"])
        
        # 递归遍历所有子目录
        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                # 不区分大小写匹配.mp4文件
                if file.lower().endswith(('.mp4', '.webm')):
                    # 构建文件路径
                    if USE_ABSOLUTE_PATH:
                        file_path = os.path.abspath(os.path.join(root, file))
                    else:
                        # 生成相对于源目录的路径
                        file_path = os.path.relpath(os.path.join(root, file), SOURCE_DIR)
                    
                    # 写入一行
                    writer.writerow([file_path])
                    count += 1
    
    print(f"✅ 生成完成！共找到 {count} 个MP4文件")
    print(f"📄 输出文件: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    # 检查源目录是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"❌ 错误：源目录不存在 - {SOURCE_DIR}")
    else:
        generate_csv()