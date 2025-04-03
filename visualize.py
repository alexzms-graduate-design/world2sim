import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

def visualize_probe_data(file_path):
    # Set the font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # Read the CSV file, skipping the header rows
    df = pd.read_csv(file_path, skiprows=4)
    
    # Extract time and probe data
    time = df.iloc[:, 0]  # First column is time
    probe1 = df.iloc[:, 1]  # Second column is probe 1
    probe2 = df.iloc[:, 2]  # Third column is probe 2
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, probe1, label='点探针1')
    plt.plot(time, probe2, label='点探针2')
    
    # Add labels and title
    plt.xlabel('时间 (s)')
    plt.ylabel('总声压 (Pa)')
    plt.title('探针数据对比')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Get the file name from user input
    file_name = input("请输入CSV文件名 (例如: output_f=250_Z=6000_T=0.1.csv): ")
    
    # Construct the full file path
    file_path = os.path.join('outputs', file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
    else:
        visualize_probe_data(file_path)
