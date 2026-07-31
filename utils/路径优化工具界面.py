#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径优化与导入导出工具界面
支持优化路径和导出多种格式，以及导入自定义理想路径
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # 用于3D投影
import json  # JSON处理
import csv  # CSV处理
import os
from datetime import datetime
import threading
from 梯度下降算法 import (
    AdamOptimizer, calculate_loss, calculate_gradients,
    export_path_to_csv, export_path_to_json, export_path_to_gcode,
    load_ideal_path_from_csv, load_ideal_path_from_json, load_ideal_path_from_gcode,
    create_sample_ideal_path
)

class PathOptimizerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("数控路径优化工具")
        self.root.geometry("1000x700")
        
        # 设置关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 初始化变量
        self.ideal_path = None
        self.initial_path = None
        self.optimized_path = None
        self.is_optimizing = False
        
        # 创建UI
        self.create_widgets()
        
        # 默认生成示例路径
        self.generate_sample_paths()
    
    def on_closing(self):
        """处理窗口关闭事件"""
        self.root.quit()  # 退出主循环
        self.root.destroy()  # 销毁窗口
        exit(0)  # 结束程序
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 路径导入区域
        import_frame = ttk.LabelFrame(control_frame, text="导入理想路径", padding="5")
        import_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(import_frame, text="从CSV导入", command=self.import_csv_path).pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="从JSON导入", command=self.import_json_path).pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="从G代码导入", command=self.import_gcode_path).pack(fill=tk.X, pady=2)
        ttk.Button(import_frame, text="生成示例路径", command=self.generate_sample_paths).pack(fill=tk.X, pady=2)
        
        # 优化参数区域
        param_frame = ttk.LabelFrame(control_frame, text="优化参数", padding="5")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.iterations_var = tk.IntVar(value=300)
        ttk.Spinbox(param_frame, from_=10, to=1000, textvariable=self.iterations_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(param_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(param_frame, from_=0.0001, to=0.01, increment=0.0001, textvariable=self.learning_rate_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(param_frame, text="噪声强度:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.noise_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(param_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.noise_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Label(param_frame, text="路径点数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.points_var = tk.IntVar(value=70)
        ttk.Spinbox(param_frame, from_=10, to=200, textvariable=self.points_var, width=10).grid(row=3, column=1, pady=2)
        
        ttk.Button(param_frame, text="开始优化", command=self.start_optimization).grid(row=4, column=0, columnspan=2, pady=5)
        
        # 导出区域
        export_frame = ttk.LabelFrame(control_frame, text="导出优化路径", padding="5")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="导出为CSV", command=lambda: self.export_path('csv')).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="导出为JSON", command=lambda: self.export_path('json')).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="导出为G代码", command=lambda: self.export_path('gcode')).pack(fill=tk.X, pady=2)
        
        # 状态区域
        status_frame = ttk.LabelFrame(control_frame, text="状态信息", padding="5")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=30, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.update_status("系统就绪。请导入理想路径或生成示例路径。")
        
        # 右侧图形区域
        plot_frame = ttk.LabelFrame(main_frame, text="路径可视化", padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig = plt.figure(figsize=(8, 6))
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132, projection='3d')
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化图表
        self.update_plots()
    
    def update_status(self, message):
        """更新状态信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        # 移除 self.root.update() 避免在子线程中更新UI
    
    def generate_sample_paths(self):
        """生成示例路径"""
        self.update_status("正在生成示例理想路径...")
        
        # 生成理想的螺旋路径
        num_points = self.points_var.get()
        z = np.linspace(0, 10, num_points)
        theta = 2 * np.pi * z / 10
        x = np.sin(theta)
        y = np.cos(theta)
        self.ideal_path = np.vstack([x, y, z]).T
        
        # 添加噪声创建初始路径
        np.random.seed(42)
        noise = np.random.normal(0, self.noise_var.get(), self.ideal_path.shape)
        self.initial_path = self.ideal_path + noise
        
        # 初始化优化路径为初始路径
        self.optimized_path = self.initial_path.copy()
        
        self.update_status(f"已生成示例路径，包含 {num_points} 个点")
        self.update_plots()
        
        # 优化路径（通过主线程调度）
        self.root.after(100, self.start_optimization)  # 延迟100ms开始优化
    
    def import_csv_path(self):
        """从CSV文件导入路径"""
        file_path = filedialog.askopenfilename(
            title="选择CSV路径文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.update_status(f"正在导入CSV文件: {os.path.basename(file_path)}")
            path = load_ideal_path_from_csv(file_path)
            
            if path is not None:
                self.ideal_path = path
                # 创建初始路径（添加噪声）
                np.random.seed(42)
                noise = np.random.normal(0, self.noise_var.get(), self.ideal_path.shape)
                self.initial_path = self.ideal_path + noise
                self.optimized_path = self.initial_path.copy()
                
                self.update_status(f"成功导入路径，包含 {len(path)} 个点")
                self.update_plots()
            else:
                messagebox.showerror("导入失败", "无法从CSV文件导入路径")
    
    def import_json_path(self):
        """从JSON文件导入路径"""
        file_path = filedialog.askopenfilename(
            title="选择JSON路径文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.update_status(f"正在导入JSON文件: {os.path.basename(file_path)}")
            path = load_ideal_path_from_json(file_path)
            
            if path is not None:
                self.ideal_path = path
                # 创建初始路径（添加噪声）
                np.random.seed(42)
                noise = np.random.normal(0, self.noise_var.get(), self.ideal_path.shape)
                self.initial_path = self.ideal_path + noise
                self.optimized_path = self.initial_path.copy()
                
                self.update_status(f"成功导入路径，包含 {len(path)} 个点")
                self.update_plots()
            else:
                messagebox.showerror("导入失败", "无法从JSON文件导入路径")
    
    def import_gcode_path(self):
        """从G代码文件导入路径"""
        file_path = filedialog.askopenfilename(
            title="选择G代码文件",
            filetypes=[("G代码文件", "*.nc,*.gcode"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.update_status(f"正在导入G代码文件: {os.path.basename(file_path)}")
            path = load_ideal_path_from_gcode(file_path)
            
            if path is not None:
                self.ideal_path = path
                # 创建初始路径（添加噪声）
                np.random.seed(42)
                noise = np.random.normal(0, self.noise_var.get(), self.ideal_path.shape)
                self.initial_path = self.ideal_path + noise
                self.optimized_path = self.initial_path.copy()
                
                self.update_status(f"成功导入路径，包含 {len(path)} 个点")
                self.update_plots()
            else:
                messagebox.showerror("导入失败", "无法从G代码文件导入路径")
    
    def start_optimization(self):
        """启动路径优化过程"""
        if self.ideal_path is None:
            messagebox.showwarning("警告", "请先导入理想路径")
            return
        
        if self.is_optimizing:
            messagebox.showinfo("提示", "优化正在进行中，请稍候...")
            return
        
        # 在新线程中运行优化，避免UI冻结
        thread = threading.Thread(target=self.optimize_path)
        thread.daemon = True
        thread.start()
    
    def optimize_path(self):
        """执行路径优化"""
        self.is_optimizing = True
        # 在主线程中更新状态
        self.root.after(0, lambda: self.update_status("开始优化路径..."))
        
        # 获取参数
        num_iterations = self.iterations_var.get()
        learning_rate = self.learning_rate_var.get()
        
        # 重置优化路径
        self.optimized_path = self.initial_path.copy()
        
        # 创建优化器
        optimizer = AdamOptimizer(self.optimized_path, lr=learning_rate)
        
        # 优化循环
        for i in range(num_iterations):
            current_loss, _, _, _ = calculate_loss(self.optimized_path, self.ideal_path)
            grad = calculate_gradients(self.optimized_path, self.ideal_path)
            self.optimized_path = optimizer.step(grad)
            
            # 定期更新UI（通过主线程）
            if i % 20 == 0:
                self.root.after(0, lambda msg=f"迭代 {i+1}/{num_iterations} - 损失: {current_loss:.4f}": self.update_status(msg))
                
                # 更新图形（每50次迭代）
                if i % 50 == 0:
                    self.root.after(0, self.update_plots)
        
        # 优化完成，在主线程中更新状态和图形
        self.root.after(0, lambda: self.update_status("优化完成！"))
        self.root.after(0, self.update_plots)
        self.is_optimizing = False
    
    def update_plots(self):
        """更新3D图形显示"""
        # 清除旧图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        if self.initial_path is not None:
            # 绘制初始路径
            self.ax1.set_title("初始噪声路径")
            self.ax1.set_xlabel("X")
            self.ax1.set_ylabel("Y")
            self.ax1.set_zlabel("Z")
            self.ax1.plot(
                self.initial_path[:, 0], 
                self.initial_path[:, 1], 
                self.initial_path[:, 2],
                'r-', marker='o', markersize=2
            )
        
        if self.optimized_path is not None:
            # 绘制优化后路径
            self.ax2.set_title("优化后路径")
            self.ax2.set_xlabel("X")
            self.ax2.set_ylabel("Y")
            self.ax2.set_zlabel("Z")
            self.ax2.plot(
                self.optimized_path[:, 0], 
                self.optimized_path[:, 1], 
                self.optimized_path[:, 2],
                'g-', marker='o', markersize=2
            )
        
        if self.ideal_path is not None:
            # 绘制理想路径
            self.ax3.set_title("理想路径")
            self.ax3.set_xlabel("X")
            self.ax3.set_ylabel("Y")
            self.ax3.set_zlabel("Z")
            self.ax3.plot(
                self.ideal_path[:, 0], 
                self.ideal_path[:, 1], 
                self.ideal_path[:, 2],
                'b-', marker='o', markersize=2
            )
        
        # 调整布局并刷新
        self.fig.tight_layout()
        self.canvas.draw()
    
    def export_path(self, format_type):
        """导出优化后的路径"""
        if self.optimized_path is None:
            messagebox.showwarning("警告", "没有可导出的优化路径")
            return
        
        # 确定默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"optimized_path_{timestamp}"
        
        # 根据格式选择扩展名
        if format_type == 'csv':
            extension = ".csv"
            default_name += extension
        elif format_type == 'json':
            extension = ".json"
            default_name += extension
        elif format_type == 'gcode':
            extension = ".nc"
            default_name += extension
        
        file_path = filedialog.asksaveasfilename(
            title="导出优化路径",
            defaultextension=extension,
            initialfile=default_name,
            filetypes=[
                (f"{format_type.upper()}文件", f"*{extension}"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                if format_type == 'csv':
                    success = export_path_to_csv(self.optimized_path, file_path)
                elif format_type == 'json':
                    metadata = {
                        "description": "优化后的数控加工路径",
                        "iterations": self.iterations_var.get(),
                        "learning_rate": self.learning_rate_var.get(),
                        "points_count": len(self.optimized_path)
                    }
                    success = export_path_to_json(self.optimized_path, metadata, file_path)
                elif format_type == 'gcode':
                    settings = {
                        "feed_rate": 200,
                        "rapid_rate": 500,
                        "comments": True
                    }
                    success = export_path_to_gcode(self.optimized_path, file_path, settings)
                
                if success:
                    self.update_status(f"成功导出为{format_type.upper()}格式: {os.path.basename(file_path)}")
                    messagebox.showinfo("导出成功", f"路径已成功导出到:\n{file_path}")
                else:
                    messagebox.showerror("导出失败", "无法导出路径")
            except Exception as e:
                messagebox.showerror("导出错误", f"导出路径时出错:\n{str(e)}")

def main():
    root = tk.Tk()
    _ = PathOptimizerUI(root)  # 创建应用实例但不直接使用
    root.mainloop()

if __name__ == "__main__":
    main()
    