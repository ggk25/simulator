import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from tkinter import filedialog

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# 导入必要的模块
from model_inference import ModelConfig, FlexibleModel, collect_model_data
from software_model.seed_oss import Prefill as SeedOssPrefill, Decode as SeedOssDecode
from software_model.deepseek import Prefill as DeepseekPrefill, Decode as DeepseekDecode
from utils import data_type_dict
from hardware_model.device import device_dict

class ModelConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("模型配置工具")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # 设置主题
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("PingFang SC", 12))
        self.style.configure("TButton", font=("PingFang SC", 12))
        self.style.configure("TCombobox", font=("PingFang SC", 12))
        self.style.configure("TRadiobutton", font=("PingFang SC", 12))
        self.style.configure("TFrame", padding="10")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建模型选择框架
        self.model_selection_frame = ttk.LabelFrame(self.main_frame, text="模型选择", padding="10")
        self.model_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模型类型选择
        self.model_type_var = tk.StringVar(value="predefined")
        ttk.Radiobutton(self.model_selection_frame, text="预定义模型", 
                       variable=self.model_type_var, value="predefined",
                       command=self.on_model_type_change).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(self.model_selection_frame, text="自定义模型", 
                       variable=self.model_type_var, value="custom",
                       command=self.on_model_type_change).pack(anchor=tk.W, pady=5)
        
        # 预定义模型选择框架
        self.predefined_frame = ttk.Frame(self.model_selection_frame)
        self.predefined_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.predefined_frame, text="选择模型:").pack(side=tk.LEFT, padx=(0, 5))
        self.predefined_model_var = tk.StringVar(value="seed_oss")
        self.predefined_model_combo = ttk.Combobox(self.predefined_frame, textvariable=self.predefined_model_var,
                                                 values=["seed_oss", "deepseek"], state="readonly")
        self.predefined_model_combo.pack(side=tk.LEFT, padx=5)
        
        # 自定义模型配置框架
        self.custom_frame = ttk.Frame(self.model_selection_frame)
        
        # 注意力机制选择
        attention_frame = ttk.Frame(self.custom_frame)
        attention_frame.pack(fill=tk.X, pady=5)
        ttk.Label(attention_frame, text="注意力机制:").pack(side=tk.LEFT, padx=(0, 5))
        self.attention_type_var = tk.StringVar(value="GQA")
        attention_combo = ttk.Combobox(attention_frame, textvariable=self.attention_type_var,
                                     values=["GQA", "MLA"], state="readonly")
        attention_combo.pack(side=tk.LEFT, padx=5)
        attention_combo.bind("<<ComboboxSelected>>", self.on_attention_type_change)
        
        # FFN类型选择
        ffn_frame = ttk.Frame(self.custom_frame)
        ffn_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ffn_frame, text="FFN类型:").pack(side=tk.LEFT, padx=(0, 5))
        self.ffn_type_var = tk.StringVar(value="FFN")
        ffn_combo = ttk.Combobox(ffn_frame, textvariable=self.ffn_type_var,
                               values=["FFN", "MoE"], state="readonly")
        ffn_combo.pack(side=tk.LEFT, padx=5)
        ffn_combo.bind("<<ComboboxSelected>>", self.on_ffn_type_change)
        
        # 创建参数配置笔记本
        self.param_notebook = ttk.Notebook(self.main_frame)
        
        # 通用参数标签页
        self.common_params_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.common_params_tab, text="通用参数")
        
        # GQA参数标签页
        self.gqa_params_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.gqa_params_tab, text="GQA参数")
        
        # MLA参数标签页
        self.mla_params_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.mla_params_tab, text="MLA参数")
        
        # FFN参数标签页
        self.ffn_params_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.ffn_params_tab, text="FFN参数")
        
        # MoE参数标签页
        self.moe_params_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.moe_params_tab, text="MoE参数")
        
        self.param_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 初始化参数输入框
        self.init_param_inputs()
        
        # 按钮框架
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        # 运行按钮
        self.run_button = ttk.Button(self.button_frame, text="运行评估", command=self.run_evaluation)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        # 导出配置按钮
        self.export_button = ttk.Button(self.button_frame, text="导出配置", command=self.export_config)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # 导入配置按钮
        self.import_button = ttk.Button(self.button_frame, text="导入配置", command=self.import_config)
        self.import_button.pack(side=tk.LEFT, padx=5)
        
        # 初始状态设置
        self.on_model_type_change()
        
    def init_param_inputs(self):
        """初始化参数输入框"""
        # 通用参数
        common_params = [
            ("模型名称:", "model_name", "qwen235B"),
            ("隐藏层大小:", "hidden_size", "4096"),
            ("数据类型:", "datatype", "fp8"),
            ("层数:", "layer_count", "94")
        ]
        
        self.common_vars = {}
        for i, (label, var_name, default) in enumerate(common_params):
            ttk.Label(self.common_params_tab, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            
            if var_name == "datatype":
                var = tk.StringVar(value=default)
                combo = ttk.Combobox(self.common_params_tab, textvariable=var, values=list(data_type_dict.keys()), state="readonly")
                combo.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            else:
                var = tk.StringVar(value=default)
                entry = ttk.Entry(self.common_params_tab, textvariable=var)
                entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            
            self.common_vars[var_name] = var
        
        # GQA参数
        gqa_params = [
            ("头维度:", "head_dim", "128"),
            ("注意力头数:", "num_attention_heads", "64"),
            ("键值头数:", "num_key_value_heads", "4"),
        ]
        
        self.gqa_vars = {}
        for i, (label, var_name, default) in enumerate(gqa_params):
            ttk.Label(self.gqa_params_tab, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(self.gqa_params_tab, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            self.gqa_vars[var_name] = var
        
        # MLA参数
        mla_params = [
            ("Q压缩维度:", "q_compress_dim", "1536"),
            ("QK RoPE维度:", "qk_rope_dim", "64"),
            ("KV压缩维度:", "kv_compress_dim", "576"),
            ("注意力头数:", "n_heads", "96"),
            ("QKV维度:", "qkv_dim", "128")
        ]
        
        self.mla_vars = {}
        for i, (label, var_name, default) in enumerate(mla_params):
            ttk.Label(self.mla_params_tab, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(self.mla_params_tab, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            self.mla_vars[var_name] = var
        
        # FFN参数
        ffn_params = [
            ("FFN中间层大小:", "ffn_intermediate_size", "11008")
        ]
        
        self.ffn_vars = {}
        for i, (label, var_name, default) in enumerate(ffn_params):
            ttk.Label(self.ffn_params_tab, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(self.ffn_params_tab, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            self.ffn_vars[var_name] = var
        
        # MoE参数
        moe_params = [
            ("专家维度:", "experts_dim", "1536"),
            ("共享专家数:", "shared_experts_count", "0"),
            ("选择的专家数:", "selected_expert_count", "8"),
            ("总专家数:", "experts_count", "128")
        ]
        
        self.moe_vars = {}
        for i, (label, var_name, default) in enumerate(moe_params):
            ttk.Label(self.moe_params_tab, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(self.moe_params_tab, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            self.moe_vars[var_name] = var
    
    def on_model_type_change(self):
        """模型类型改变时的处理"""
        if self.model_type_var.get() == "predefined":
            self.predefined_frame.pack(fill=tk.X, pady=5)
            self.custom_frame.pack_forget()
            self.param_notebook.pack_forget()
        else:
            self.predefined_frame.pack_forget()
            self.custom_frame.pack(fill=tk.X, pady=5)
            self.param_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.on_attention_type_change()
            self.on_ffn_type_change()
    
    def on_attention_type_change(self, event=None):
        """注意力类型改变时的处理"""
        if self.attention_type_var.get() == "GQA":
            self.param_notebook.tab(self.gqa_params_tab, state="normal")
            self.param_notebook.tab(self.mla_params_tab, state="disabled")
            self.param_notebook.select(self.gqa_params_tab)
        else:
            self.param_notebook.tab(self.gqa_params_tab, state="disabled")
            self.param_notebook.tab(self.mla_params_tab, state="normal")
            self.param_notebook.select(self.mla_params_tab)
    
    def on_ffn_type_change(self, event=None):
        """FFN类型改变时的处理"""
        if self.ffn_type_var.get() == "FFN":
            self.param_notebook.tab(self.ffn_params_tab, state="normal")
            self.param_notebook.tab(self.moe_params_tab, state="disabled")
            self.param_notebook.select(self.ffn_params_tab)
        else:
            self.param_notebook.tab(self.ffn_params_tab, state="disabled")
            self.param_notebook.tab(self.moe_params_tab, state="normal")
            self.param_notebook.select(self.moe_params_tab)
    
    def get_model_config(self):
        """获取模型配置"""
        if self.model_type_var.get() == "predefined":
            # 返回预定义模型配置
            model_name = self.predefined_model_var.get()
            datatype = data_type_dict["fp8"]
            
            if model_name == "seed_oss":
                return {
                    'name': model_name,
                    'prefill_class': SeedOssPrefill,
                    'decode_class': SeedOssDecode,
                    'hidden_size': 5120,
                    'datatype': datatype
                }
            elif model_name == "deepseek":
                return {
                    'name': model_name,
                    'prefill_class': DeepseekPrefill,
                    'decode_class': DeepseekDecode,
                    'hidden_size': 7168,
                    'datatype': datatype
                }
        else:
            # 返回自定义模型配置
            config = {
                'name': self.common_vars['model_name'].get(),
                'attention_type': self.attention_type_var.get(),
                'ffn_type': self.ffn_type_var.get(),
                'hidden_size': int(self.common_vars['hidden_size'].get()),
                'datatype': data_type_dict[self.common_vars['datatype'].get()],
                'layer_count': int(self.common_vars['layer_count'].get())
            }
            
            # 添加GQA参数
            if config['attention_type'] == "GQA":
                for var_name, var in self.gqa_vars.items():
                    config[var_name] = int(var.get())
            
            # 添加MLA参数
            if config['attention_type'] == "MLA":
                for var_name, var in self.mla_vars.items():
                    config[var_name] = int(var.get())
            
            # 添加FFN参数
            if config['ffn_type'] == "FFN":
                for var_name, var in self.ffn_vars.items():
                    config[var_name] = int(var.get())
            
            # 添加MoE参数
            if config['ffn_type'] == "MoE":
                for var_name, var in self.moe_vars.items():
                    config[var_name] = int(var.get())
            
            return ModelConfig(**config)
    
    def run_evaluation(self):
        """运行模型评估"""
        try:
            # 获取模型配置
            model_config = self.get_model_config()
            
            # 显示进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("运行中")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="正在运行模型评估...", font=("PingFang SC", 12)).pack(pady=20)
            self.root.update_idletasks()
            
            # 配置评估参数
            device = device_dict["D37x"]
            batch_size = 256
            prefill_length = 1024
            decode_length = 2048
            
            # 收集模型数据
            use_flexible_model = isinstance(model_config, ModelConfig)
            model_data = collect_model_data(
                model_config,
                device=device,
                batch_size=batch_size,
                prefill_lenth=prefill_length,
                decode_lenth=decode_length,
                use_flexible_model=use_flexible_model
            )
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示结果
            self.show_results(model_data)
            
        except Exception as e:
            messagebox.showerror("错误", f"运行评估时出错: {str(e)}")
    
    def show_results(self, model_data):
        """显示评估结果"""
        result_window = tk.Toplevel(self.root)
        result_window.title("评估结果")
        result_window.geometry("800x600")
        
        # 创建表格框架
        table_frame = ttk.Frame(result_window)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建表格
        columns = ("阶段", "算子", "总延时(ms)", "算子占比")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充数据
        for item in model_data:
            if item:  # 跳过空行
                values = (
                    item.get('阶段', ''),
                    item.get('算子', ''),
                    f"{item.get('总延时(ms)', 0):.2f}",
                    f"{item.get('算子占总延时比例', 0):.2%}"
                )
                tree.insert("", tk.END, values=values)
        
        # 添加导出按钮
        export_button = ttk.Button(result_window, text="导出结果", 
                                command=lambda: self.export_results(model_data))
        export_button.pack(pady=10)
    
    def export_results(self, model_data):
        """导出评估结果"""
        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if file_path:
                # 创建DataFrame
                df = pd.DataFrame(model_data)
                
                # 导出到Excel
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    # 1. 延迟数据工作表 - 只包含A-E列（模型、阶段、算子、总延时、算子占比）
                    if '模型' in df.columns and '阶段' in df.columns and '算子' in df.columns and '总延时(ms)' in df.columns and '算子占总延时比例' in df.columns:
                        latency_df = df[['模型', '阶段', '算子', '总延时(ms)', '算子占总延时比例']].copy()
                        latency_df.to_excel(writer, sheet_name='延迟数据', index=False)
                    else:
                        # 如果缺少某些列，使用可用的前5列
                        available_cols = df.columns[:5] if len(df.columns) >= 5 else df.columns
                        latency_df = df[available_cols].copy()
                        latency_df.to_excel(writer, sheet_name='延迟数据', index=False)
                    
                    # 2. 汇总数据工作表 - 包含首词延迟、总延时和吞吐率等汇总数据
                    summary_data = []
                    
                    # 提取汇总数据
                    prefill_summary = None
                    decode_summary = None
                    
                    for item in model_data:
                        if item and item.get('算子') == '汇总':
                            if item.get('阶段') == 'prefill':
                                prefill_summary = item
                            elif item.get('阶段') == 'decode':
                                decode_summary = item
                    
                    # 获取模型名称
                    model_name = model_data[0].get('模型', '未知模型') if model_data else '未知模型'
                    
                    # 构建汇总数据
                    if prefill_summary and decode_summary:
                        summary_data.append({
                            '模型': model_name,
                            '首词延迟(TTFT)(s)': prefill_summary.get('TTFT(s)', 0),
                            '总延时(ms)': prefill_summary.get('总延时(ms)', 0) + decode_summary.get('总延时(ms)', 0),
                            '吞吐率(tokens/s)': decode_summary.get('吞吐率(tokens/s)', 0),
                            '生成速度(tokens/s)': decode_summary.get('生成速度(tokens/s)', 0),
                            'Prefill计算延时(ms)': prefill_summary.get('计算延时(ms)', 0),
                            'Prefill通信延时(ms)': prefill_summary.get('通信延时(ms)', 0),
                            'Decode计算延时(ms)': decode_summary.get('计算延时(ms)', 0),
                            'Decode通信延时(ms)': decode_summary.get('通信延时(ms)', 0)
                        })
                    else:
                        # 如果没有完整的汇总数据，提供基本汇总
                        summary_data.append({
                            '模型': model_name,
                            '首词延迟(TTFT)(s)': 0,
                            '总延时(ms)': 0,
                            '吞吐率(tokens/s)': 0,
                            '生成速度(tokens/s)': 0,
                            'Prefill计算延时(ms)': 0,
                            'Prefill通信延时(ms)': 0,
                            'Decode计算延时(ms)': 0,
                            'Decode通信延时(ms)': 0
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
                    
                    # 获取工作表对象并设置格式
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        
                        # 设置列宽
                        if sheet_name == '延迟数据':
                            worksheet.set_column('A:A', 15)  # 模型
                            worksheet.set_column('B:B', 10)  # 阶段
                            worksheet.set_column('C:C', 30)  # 算子
                            worksheet.set_column('D:E', 15)  # 总延时和比例
                        elif sheet_name == '汇总数据':
                            worksheet.set_column('A:A', 15)  # 模型
                            worksheet.set_column('B:I', 15)  # 其他列
                
                messagebox.showinfo("成功", f"结果已导出到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出结果时出错: {str(e)}")
    
    def export_config(self):
        """导出模型配置"""
        try:
            # 获取模型配置
            model_config = self.get_model_config()
            
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                import json
                
                # 如果是ModelConfig对象，转换为字典
                if isinstance(model_config, ModelConfig):
                    config_dict = {
                        'name': model_config.name,
                        'attention_type': model_config.attention_type,
                        'ffn_type': model_config.ffn_type,
                        'hidden_size': model_config.hidden_size,
                        'layer_count': model_config.layer_count
                    }
                    
                    # 添加GQA参数
                    if model_config.attention_type == "GQA":
                        config_dict.update({
                            'head_dim': model_config.head_dim,
                            'num_attention_heads': model_config.num_attention_heads,
                            'num_key_value_heads': model_config.num_key_value_heads,
                            'intermediate_size': model_config.intermediate_size
                        })
                    
                    # 添加MLA参数
                    if model_config.attention_type == "MLA":
                        config_dict.update({
                            'q_compress_dim': model_config.q_compress_dim,
                            'qk_rope_dim': model_config.qk_rope_dim,
                            'kv_compress_dim': model_config.kv_compress_dim,
                            'n_heads': model_config.n_heads,
                            'qkv_dim': model_config.qkv_dim
                        })
                    
                    # 添加FFN参数
                    if model_config.ffn_type == "FFN":
                        config_dict.update({
                            'ffn_intermediate_size': model_config.ffn_intermediate_size
                        })
                    
                    # 添加MoE参数
                    if model_config.ffn_type == "MoE":
                        config_dict.update({
                            'experts_dim': model_config.experts_dim,
                            'shared_experts_count': model_config.shared_experts_count,
                            'selected_expert_count': model_config.selected_expert_count,
                            'experts_count': model_config.experts_count
                        })
                else:
                    # 预定义模型配置
                    config_dict = {
                        'name': model_config['name'],
                        'type': 'predefined'
                    }
                
                # 保存到文件
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                
                messagebox.showinfo("成功", f"配置已导出到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出配置时出错: {str(e)}")
    
    def import_config(self):
        """导入模型配置"""
        try:
            # 选择文件路径
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                import json
                
                # 读取配置文件
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
                
                # 如果是预定义模型
                if config_dict.get('type') == 'predefined':
                    self.model_type_var.set("predefined")
                    self.predefined_model_var.set(config_dict['name'])
                else:
                    # 自定义模型
                    self.model_type_var.set("custom")
                    
                    # 设置通用参数
                    self.common_vars['model_name'].set(config_dict.get('name', ''))
                    self.common_vars['hidden_size'].set(config_dict.get('hidden_size', ''))
                    self.common_vars['layer_count'].set(config_dict.get('layer_count', ''))
                    
                    # 设置注意力类型和参数
                    attention_type = config_dict.get('attention_type', 'GQA')
                    self.attention_type_var.set(attention_type)
                    
                    if attention_type == "GQA":
                        for var_name, var in self.gqa_vars.items():
                            var.set(config_dict.get(var_name, ''))
                    elif attention_type == "MLA":
                        for var_name, var in self.mla_vars.items():
                            var.set(config_dict.get(var_name, ''))
                    
                    # 设置FFN类型和参数
                    ffn_type = config_dict.get('ffn_type', 'FFN')
                    self.ffn_type_var.set(ffn_type)
                    
                    if ffn_type == "FFN":
                        for var_name, var in self.ffn_vars.items():
                            var.set(config_dict.get(var_name, ''))
                    elif ffn_type == "MoE":
                        for var_name, var in self.moe_vars.items():
                            var.set(config_dict.get(var_name, ''))
                
                # 更新界面
                self.on_model_type_change()
                self.on_attention_type_change()
                self.on_ffn_type_change()
                
                messagebox.showinfo("成功", f"配置已从 {file_path} 导入")
        except Exception as e:
            messagebox.showerror("错误", f"导入配置时出错: {str(e)}")

def main():
    root = tk.Tk()
    app = ModelConfigGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()