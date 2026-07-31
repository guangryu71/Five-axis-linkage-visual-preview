import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import csv
import os
import re
from datetime import datetime
# 设置中文字体（使用系统已安装的中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
    
    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.params -= update
        return self.params

def calculate_loss(path, original_path):
    """改进的损失函数，添加形状保持约束"""
    diffs = np.diff(path, axis=0)
    
    # 1. 路径长度损失
    path_length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    
    # 2. 平滑度损失 (二阶差分)
    second_diffs = np.diff(path, n=2, axis=0)
    smoothness = np.sum(np.sum(second_diffs ** 2, axis=1))
    
    # 3. 形状保持损失 (与原始螺旋路径的相似度)
    radial_dist = np.sqrt(path[:,0]**2 + path[:,1]**2)  # 当前径向距离
    original_radial = np.sqrt(original_path[:,0]**2 + original_path[:,1]**2)
    shape_loss = np.sum((radial_dist - original_radial)**2)  # 保持螺旋半径
    
    # 平衡各项损失权重
    # 如果形状保持是最高优先级，可以这样调整：
    total_loss = path_length + 0.01*smoothness + 1.0*shape_loss

    return total_loss, path_length, smoothness, shape_loss

def calculate_gradients(path, original_path, delta=1e-5):
    """改进的梯度计算"""
    grad = np.zeros_like(path)
    #？？？
    original_loss, _, _, _ = calculate_loss(path, original_path)
    
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            original_val = path[i, j]
            
            # 正向扰动
            path[i, j] = original_val + delta
            loss_plus, _, _, _ = calculate_loss(path, original_path)
            
            # 负向扰动
            path[i, j] = original_val - delta
            loss_minus, _, _, _ = calculate_loss(path, original_path)
            
            path[i, j] = original_val
            grad[i, j] = (loss_plus - loss_minus) / (2 * delta)
    
    return grad

def plot_3d_path(ax, path, title, color='blue'):
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 
            marker='o', linestyle='-', color=color, markersize=4)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True)

# 生成理想的螺旋路径作为参考
num_points = 70
z = np.linspace(0, 10, num_points)
theta = 2 * np.pi * z / 10
x = np.sin(theta)
y = np.cos(theta)
ideal_path = np.vstack([x, y, z]).T

# 添加适量噪声创建初始路径
np.random.seed(42)
noise = np.random.normal(0, 0.1, ideal_path.shape)
initial_path = ideal_path + noise

# 优化参数
num_iterations = 450
learning_rate = 0.00080
optimized_path = initial_path.copy()

# 创建优化器
optimizer = AdamOptimizer(optimized_path, lr=learning_rate)

# 存储损失历史
loss_history = []
path_history = [initial_path.copy()]

# 优化循环
for i in range(num_iterations):
    current_loss, length_loss, smooth_loss, shape_loss = calculate_loss(optimized_path, ideal_path)
    grad = calculate_gradients(optimized_path, ideal_path)
    optimized_path = optimizer.step(grad)
    loss_history.append(current_loss)
    
    if i % 50 == 0 or i == num_iterations - 1:
        path_history.append(optimized_path.copy())
    
    if i % 20 == 0:
        print(f"Iteration {i:3d} | Loss: {current_loss:.4f} | "
              f"Length: {length_loss:.4f} | Smooth: {smooth_loss:.4f} | Shape: {shape_loss:.4f}")

print("完成！")

# ==================== 路径导出功能 ====================

def export_path_to_csv(path, filename=None):
    """将路径导出为CSV文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_path_{timestamp}.csv"
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y', 'Z'])  # CSV头部
            writer.writerows(path)  # 写入路径点
        
        print(f"路径已成功导出到: {filename}")
        return True
    except Exception as e:
        print(f"导出路径到CSV失败: {e}")
        return False

def export_path_to_json(path, metadata=None, filename=None):
    """将路径导出为JSON文件，包含额外的元数据"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_path_{timestamp}.json"
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    # 创建要导出的数据结构
    export_data = {
        "metadata": metadata or {
            "created": datetime.now().isoformat(),
            "description": "优化后的数控加工路径",
            "points_count": len(path)
        },
        "path": path.tolist()
    }
    
    try:
        with open(filename, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=4)
        
        print(f"路径已成功导出到: {filename}")
        return True
    except Exception as e:
        print(f"导出路径到JSON失败: {e}")
        return False

def export_path_to_gcode(path, filename=None, settings=None):
    """将路径导出为G代码格式"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_path_{timestamp}.nc"
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    # 默认G代码设置
    default_settings = {
        "feed_rate": 200,  # mm/min
        "rapid_rate": 500,  # mm/min
        "z_height_safe": 50,  # mm
        "z_start": 30,  # mm
        "comments": True
    }
    
    # 合并用户设置和默认设置
    gcode_settings = {**default_settings, **(settings or {})}
    
    try:
        with open(filename, 'w') as gcode_file:
            # 写入G代码头部
            gcode_file.write("; 优化后的数控加工路径\n")
            gcode_file.write(f"; 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            gcode_file.write(f"; 路径点数: {len(path)}\n\n")
            
            # 初始化
            gcode_file.write("G21 ; 设置单位为毫米\n")
            gcode_file.write("G90 ; 绝对坐标编程\n")
            gcode_file.write("G94 ; 进给率单位为毫米/分钟\n\n")
            
            # 移动到安全高度和起点
            first_point = path[0]
            gcode_file.write(f"G0 X{first_point[0]:.3f} Y{first_point[1]:.3f} Z{gcode_settings['z_height_safe']:.3f} ; 快速移动到起点上方\n")
            gcode_file.write(f"G1 Z{gcode_settings['z_start']:.3f} F{gcode_settings['rapid_rate']} ; 下移到加工起始高度\n\n")
            
            # 写入路径点
            for i, point in enumerate(path):
                if i == 0:
                    continue  # 第一个点已经处理过了
                
                prev_point = path[i-1]
                curr_point = point
                
                # 判断是否需要抬刀（如切削结束到下一个切削开始）
                z_change = abs(curr_point[2] - prev_point[2])
                
                if z_change > 5:  # 如果Z轴变化大于5mm，抬刀
                    gcode_file.write(f"G0 Z{gcode_settings['z_height_safe']:.3f} ; 快速抬刀到安全高度\n")
                    gcode_file.write(f"G0 X{curr_point[0]:.3f} Y{curr_point[1]:.3f} ; 快速移动到下一个点上方\n")
                    gcode_file.write(f"G1 Z{curr_point[2]:.3f} F{gcode_settings['rapid_rate']} ; 下移到加工高度\n")
                else:
                    gcode_file.write(f"G1 X{curr_point[0]:.3f} Y{curr_point[1]:.3f} Z{curr_point[2]:.3f} F{gcode_settings['feed_rate']} ; 切削移动\n")
                
                # 添加注释（如果启用）
                if gcode_settings["comments"] and i % 10 == 0:
                    gcode_file.write(f"; 路径点 {i+1}/{len(path)}\n")
            
            # 结束 - 抬刀并返回原点
            gcode_file.write("\nG0 Z50 ; 快速抬刀\n")
            gcode_file.write("G28 ; 返回原点\n")
            gcode_file.write("M30 ; 程序结束\n")
        
        print(f"路径已成功导出为G代码: {filename}")
        return True
    except Exception as e:
        print(f"导出路径到G代码失败: {e}")
        return False

# 导出优化路径（示例）
# export_path_to_csv(optimized_path, "../output/optimized_path.csv")
# export_path_to_json(optimized_path, {"description": "五轴联动优化路径", "iterations": num_iterations}, "../output/optimized_path.json")
# export_path_to_gcode(optimized_path, "../output/optimized_path.nc", {"feed_rate": 150})

# ==================== 理想路径导入功能 ====================

def load_ideal_path_from_csv(filename):
    """从CSV文件导入理想路径"""
    try:
        path = []
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 跳过标题行
            
            for row in reader:
                if len(row) >= 3:
                    point = [float(row[0]), float(row[1]), float(row[2])]
                    path.append(point)
        
        print(f"成功从CSV文件导入路径: {filename} ({len(path)}个点)")
        return np.array(path)
    except Exception as e:
        print(f"从CSV导入路径失败: {e}")
        return None

def load_ideal_path_from_json(filename):
    """从JSON文件导入理想路径"""
    try:
        with open(filename, 'r') as jsonfile:
            data = json.load(jsonfile)
        
        # 检查路径数据是否存在
        if "path" in data:
            path = np.array(data["path"])
            print(f"成功从JSON文件导入路径: {filename} ({len(path)}个点)")
            
            # 打印元数据（如果有）
            if "metadata" in data and "description" in data["metadata"]:
                print(f"路径描述: {data['metadata']['description']}")
            
            return path
        else:
            print("JSON文件中未找到路径数据")
            return None
    except Exception as e:
        print(f"从JSON导入路径失败: {e}")
        return None

def load_ideal_path_from_gcode(filename, extract_z_changes=True):
    """从G代码文件导入理想路径"""
    try:
        path = []
        with open(filename, 'r') as gcode_file:
            for line in gcode_file:
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith(';'):
                    continue
                
                # 解析G1或G0移动指令
                if line.startswith(('G1', 'G0', 'G01', 'G00')):
                    x, y, z = None, None, None
                    
                    # 使用正则表达式提取坐标值
                    x_match = re.search(r'X([-\\d.]+)', line)
                    y_match = re.search(r'Y([-\\d.]+)', line)
                    z_match = re.search(r'Z([-\\d.]+)', line)
                    
                    if x_match:
                        x = float(x_match.group(1))
                    if y_match:
                        y = float(y_match.group(1))
                    if z_match:
                        z = float(z_match.group(1))
                    
                    # 如果有Z轴坐标或者提取所有变化，则添加到路径
                    if z is not None or not extract_z_changes:
                        # 使用最后一个已知坐标或默认为0
                        if x is None and path:
                            x = path[-1][0]
                        elif x is None:
                            x = 0
                            
                        if y is None and path:
                            y = path[-1][1]
                        elif y is None:
                            y = 0
                            
                        if z is None and path:
                            z = path[-1][2]
                        elif z is None:
                            z = 0
                        
                        path.append([x, y, z])
        
        path = np.array(path)
        print(f"成功从G代码文件导入路径: {filename} ({len(path)}个点)")
        return path
    except Exception as e:
        print(f"从G代码导入路径失败: {e}")
        return None

def create_sample_ideal_path(output_dir="../output"):
    """创建一个示例理想路径文件用于测试"""
    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成理想的螺旋路径作为示例
    num_points = 100
    z = np.linspace(0, 20, num_points)
    theta = 2 * np.pi * z / 10
    x = np.sin(theta) * 10
    y = np.cos(theta) * 10
    sample_path = np.vstack([x, y, z]).T
    
    # 导出为不同格式
    export_path_to_csv(sample_path, f"{output_dir}/ideal_spiral_path.csv")
    export_path_to_json(sample_path, {"description": "理想螺旋路径示例"}, f"{output_dir}/ideal_spiral_path.json")
    export_path_to_gcode(sample_path, f"{output_dir}/ideal_spiral_path.nc", {"feed_rate": 150})
    
    return sample_path

# 示例用法：
# 从不同格式导入理想路径：
# ideal_path_csv = load_ideal_path_from_csv("ideal_path.csv")
# ideal_path_json = load_ideal_path_from_json("ideal_path.json")
# ideal_path_gcode = load_ideal_path_from_gcode("ideal_path.nc")

# 创建示例理想路径文件：
# sample_path = create_sample_ideal_path()