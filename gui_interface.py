import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from hardware_model.device import Device
from hardware_model.compute_module import VectorUnit, SystolicArray, compute_module
from hardware_model.memory_module import MemoryModule
from hardware_model.communication import onchip_communacation, scale_up, scale_out, communication
from software_model.seed_oss import Prefill as SeedOssPrefill, Decode as SeedOssDecode
from software_model.deepseek import Prefill as DeepseekPrefill, Decode as DeepseekDecode
from utils import DataType, Tensor, data_type_dict
from model_inference.integration_to_excel import collect_model_data

# 设置中文字体 - 适用于macOS系统
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI显示问题
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PerformanceModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("模型性能评估工具")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # 设置主题
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("PingFang SC", 13))
        self.style.configure("TButton", font=("PingFang SC", 13))
        self.style.configure("TCombobox", font=("PingFang SC", 13))
        self.style.configure("TCheckbutton", font=("PingFang SC", 13))
        self.style.configure("TRadiobutton", font=("PingFang SC", 13))

        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20 20 20 20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建参数配置笔记本
        self.tab_control = ttk.Notebook(self.main_frame)

        # 模型和基本参数标签页
        self.basic_params_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.basic_params_tab, text="模型和基本参数")

        # 硬件参数标签页
        self.hardware_params_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.hardware_params_tab, text="硬件参数配置")

        self.tab_control.pack(expand=1, fill="both", pady=(0, 10))

        # ========== 模型和基本参数标签页 ==========
        # 模型选择
        ttk.Label(self.basic_params_tab, text="推理模型选择:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="seed_oss")
        model_frame = ttk.Frame(self.basic_params_tab)
        model_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(model_frame, text="Seed OSS", variable=self.model_var, value="seed_oss").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="DeepSeek", variable=self.model_var, value="deepseek").pack(side=tk.LEFT, padx=5)

        # Prompt长度
        ttk.Label(self.basic_params_tab, text="Prompt长度:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.prompt_var = tk.StringVar(value="1024")
        ttk.Entry(self.basic_params_tab, textvariable=self.prompt_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Decode生成长度
        ttk.Label(self.basic_params_tab, text="Decode生成长度:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.decode_var = tk.StringVar(value="2048")
        ttk.Entry(self.basic_params_tab, textvariable=self.decode_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # 批处理大小
        ttk.Label(self.basic_params_tab, text="批处理大小:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_var = tk.StringVar(value="256")
        ttk.Entry(self.basic_params_tab, textvariable=self.batch_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # 数据类型
        ttk.Label(self.basic_params_tab, text="数据类型:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.dtype_var = tk.StringVar(value="fp8")
        dtype_frame = ttk.Frame(self.basic_params_tab)
        dtype_frame.grid(row=4, column=1, sticky=tk.W, pady=5)
        for dtype in data_type_dict.keys():
            ttk.Radiobutton(dtype_frame, text=dtype, variable=self.dtype_var, value=dtype).pack(side=tk.LEFT, padx=5)

        # ========== 硬件参数标签页 ==========
        # 创建滚动框架
        self.hardware_canvas = tk.Canvas(self.hardware_params_tab)
        self.hardware_scrollbar = ttk.Scrollbar(self.hardware_params_tab, orient="vertical", command=self.hardware_canvas.yview)
        self.hardware_frame = ttk.Frame(self.hardware_canvas)

        self.hardware_frame.bind(
            "<Configure>",
            lambda e: self.hardware_canvas.configure(scrollregion=self.hardware_canvas.bbox("all"))
        )

        self.hardware_canvas.create_window((0, 0), window=self.hardware_frame, anchor="nw")
        self.hardware_canvas.configure(yscrollcommand=self.hardware_scrollbar.set)

        self.hardware_canvas.pack(side="left", fill="both", expand=True)
        self.hardware_scrollbar.pack(side="right", fill="y")

        # 基本硬件参数
        ttk.Label(self.hardware_frame, text="=== 基本硬件参数 ===", font=("PingFang SC", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        # PE数量
        ttk.Label(self.hardware_frame, text="PE数量:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.pe_count_var = tk.StringVar(value="4")
        ttk.Entry(self.hardware_frame, textvariable=self.pe_count_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # 核心数量
        ttk.Label(self.hardware_frame, text="核心数量:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.core_count_var = tk.StringVar(value="15")
        ttk.Entry(self.hardware_frame, textvariable=self.core_count_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # 频率(MHz)
        ttk.Label(self.hardware_frame, text="频率(MHz):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.frequency_var = tk.StringVar(value="800")
        ttk.Entry(self.hardware_frame, textvariable=self.frequency_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # 内存模块参数
        ttk.Label(self.hardware_frame, text="=== 内存模块参数 ===", font=("PingFang SC", 13, "bold")).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        # DRAM带宽(GB/s)
        ttk.Label(self.hardware_frame, text="DRAM带宽(GB/s):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.dram_bandwidth_var = tk.StringVar(value="12288")
        ttk.Entry(self.hardware_frame, textvariable=self.dram_bandwidth_var, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

        # DRAM带宽利用率(%)
        ttk.Label(self.hardware_frame, text="DRAM带宽利用率(%):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.dram_util_var = tk.StringVar(value="0.8")
        ttk.Entry(self.hardware_frame, textvariable=self.dram_util_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)

        # 通信模块参数
        ttk.Label(self.hardware_frame, text="=== 通信模块参数 ===", font=("PingFang SC", 13, "bold")).grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        # NoC带宽(GB/s)
        ttk.Label(self.hardware_frame, text="NoC带宽(GB/s):").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.noc_bandwidth_var = tk.StringVar(value="128")
        ttk.Entry(self.hardware_frame, textvariable=self.noc_bandwidth_var, width=10).grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        
        # NoC链路延迟(cycles)
        self.noc_link_latency_var = tk.StringVar(value="30")

        # NoC路由延迟(cycles)
        self.noc_router_latency_var = tk.StringVar(value="3")

        # scale_out带宽(GB/s)
        ttk.Label(self.hardware_frame, text="卡间互联带宽(GB/s):").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.scale_out_bandwidth_var = tk.StringVar(value="128")
        ttk.Entry(self.hardware_frame, textvariable=self.scale_out_bandwidth_var, width=10).grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)

        # 计算模块参数
        ttk.Label(self.hardware_frame, text="=== 计算模块参数 ===", font=("PingFang SC", 13, "bold")).grid(row=11, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

        # 向量宽度
        ttk.Label(self.hardware_frame, text="向量宽度:").grid(row=12, column=0, sticky=tk.W, padx=5, pady=5)
        self.vector_width_var = tk.StringVar(value="32")
        ttk.Entry(self.hardware_frame, textvariable=self.vector_width_var, width=10).grid(row=12, column=1, sticky=tk.W, padx=5, pady=5)

        # 阵列高度
        ttk.Label(self.hardware_frame, text="阵列高度:").grid(row=13, column=0, sticky=tk.W, padx=5, pady=5)
        self.array_height_var = tk.StringVar(value="32")
        ttk.Entry(self.hardware_frame, textvariable=self.array_height_var, width=10).grid(row=13, column=1, sticky=tk.W, padx=5, pady=5)

        # 阵列宽度
        ttk.Label(self.hardware_frame, text="阵列宽度:").grid(row=14, column=0, sticky=tk.W, padx=5, pady=5)
        self.array_width_var = tk.StringVar(value="32")
        ttk.Entry(self.hardware_frame, textvariable=self.array_width_var, width=10).grid(row=14, column=1, sticky=tk.W, padx=5, pady=5)

        # 每周期MAC数
        ttk.Label(self.hardware_frame, text="每周期MAC数:").grid(row=15, column=0, sticky=tk.W, padx=5, pady=5)
        self.mac_per_cycle_var = tk.StringVar(value="1")
        ttk.Entry(self.hardware_frame, textvariable=self.mac_per_cycle_var, width=10).grid(row=15, column=1, sticky=tk.W, padx=5, pady=5)

        # 运行按钮
        self.run_button = ttk.Button(self.main_frame, text="运行评估并导出到Excel", command=self.run_evaluation)
        self.run_button.pack(pady=10)

        # 初始化数据
        self.result_data = None
        self.device_info = None
        self.model_info = None

    def run_evaluation(self):
        """运行性能评估并导出到Excel"""
        try:
            # 禁用运行按钮
            self.run_button.config(state=tk.DISABLED)
            self.root.update_idletasks()

            # 显示加载提示
            loading_window = tk.Toplevel(self.root)
            loading_window.title("运行中")
            loading_window.geometry("250x100")
            loading_window.transient(self.root)
            loading_window.grab_set()
            ttk.Label(loading_window, text="正在计算性能指标并导出到Excel...", font=("PingFang SC", 12)).pack(pady=20)
            self.root.update_idletasks()

            # 获取模型和基本参数
            model_name = self.model_var.get()
            prefill_lenth = int(self.prompt_var.get())
            decode_lenth = int(self.decode_var.get())
            batch_size = int(self.batch_var.get())
            datatype = data_type_dict[self.dtype_var.get()]

            # 获取硬件参数
            pe_count = int(self.pe_count_var.get())
            core_count = int(self.core_count_var.get())
            frequency = int(self.frequency_var.get())
            dram_bandwidth = float(self.dram_bandwidth_var.get())
            dram_util = float(self.dram_util_var.get())
            noc_bandwidth = float(self.noc_bandwidth_var.get())
            noc_link_latency = int(self.noc_link_latency_var.get())
            noc_router_latency = int(self.noc_router_latency_var.get())
            scale_out_bandwidth = float(self.scale_out_bandwidth_var.get())
            vector_width = int(self.vector_width_var.get())
            array_height = int(self.array_height_var.get())
            array_width = int(self.array_width_var.get())
            mac_per_cycle = int(self.mac_per_cycle_var.get())

            # 创建硬件组件
            vector_unit = VectorUnit(4, 15, 12, 15, 2, vector_width)
            systolic_array = SystolicArray(array_height, array_width, mac_per_cycle, 1, 4)
            compute_module_obj = compute_module(vector_unit, systolic_array)
            memory_module = MemoryModule(dram_bandwidth, dram_util)
            onchip_comm = onchip_communacation(noc_link_latency, noc_router_latency, noc_bandwidth, frequency)
            scale_up_comm = scale_up(50, 2*scale_out_bandwidth, frequency)
            scale_out_comm = scale_out(100, scale_out_bandwidth, frequency)
            communication_obj = communication(onchip_comm, scale_up_comm, scale_out_comm)

            # 创建设备
            device = Device(
                compute_module=compute_module_obj,
                memory=memory_module,
                communication=communication_obj,
                PE_count=pe_count,
                core_count=core_count,
                frequency=frequency
            )

            # 记录硬件信息
            self.device_info = {
                "设备名称": "自定义配置",
                "PE数量": pe_count,
                "核心数量": core_count,
                "频率(MHz)": frequency,
                "DRAM带宽(GB/s)": dram_bandwidth,
                "DRAM带宽利用率": dram_util,
                "NoC带宽(GB/s)": noc_bandwidth,
                "卡间互联带宽(GB/s)": scale_out_bandwidth,
                "向量宽度": vector_width,
                "阵列高度": array_height,
                "阵列宽度": array_width,
                "每周期MAC数": mac_per_cycle,
                "矩阵乘算力@4bits(tflops)": 2 * mac_per_cycle * array_height * array_width * pe_count * core_count * frequency*2/1e6,
                "矩阵乘算力@8bits(tflops)": mac_per_cycle * array_height * array_width * pe_count * core_count * frequency*2/1e6,
                "矩阵乘算力@16bits(tflops)": mac_per_cycle * array_height * array_width * pe_count * core_count * frequency/1e6,
            }

            # 根据模型选择加载相应的类
            if model_name == "seed_oss":
                prefill_class = SeedOssPrefill
                decode_class = SeedOssDecode
                hidden_size = 5120
                self.model_info = {
                    "模型名称": "Seed OSS",
                    "参数量": "36B",
                    "层数": 64,
                    "模型类型": "GQA + dense FFN",
                    "隐藏大小": hidden_size,
                    "头维度": 128,
                    "注意力头数": 80,
                    "键值头数": 8,
                    "中间层大小": 27648
                }
            else:  # deepseek
                prefill_class = DeepseekPrefill
                decode_class = DeepseekDecode
                hidden_size = 7168
                self.model_info = {
                    "模型名称": "DeepSeek",
                    "参数量": "671B", 
                    "层数": 61,
                    "模型类型": "MLA + MoE",
                    "隐藏大小": hidden_size,
                    "查询压缩维度": 1536,
                    "RoPE维度": 64,
                    "键值压缩维度": 576,
                    "头数": 128,
                    "查询键值维度": 128,
                    "专家维度": 2048,
                    "选中专家数": 8,
                    "专家总数": 256
                }
            if model_name == "deepseek":
                compute_graph_data = [
                    # Prefill阶段
                    {"阶段": "Prefill", "算子": "mla rmsnorm", "计算维度": "(b,s,7168)", "前驱算子": "输入", "后继算子": "linear(q_a)"},
                    {"阶段": "Prefill", "算子": "linear(q_a)", "计算维度": "(b,s,7168)*(7168,1536)", "前驱算子": "mla rmsnorm", "后继算子": "rmsnorm(q)"},
                    {"阶段": "Prefill", "算子": "rmsnorm(q)", "计算维度": "(b,s,1536)", "前驱算子": "linear(q_a)", "后继算子": "linear(q_b)"},
                    {"阶段": "Prefill", "算子": "linear(q_b)", "计算维度": "(b,s,1536)*(1536,16384)", "前驱算子": "rmsnorm(q)", "后继算子": "rope_q"},
                    {"阶段": "Prefill", "算子": "rope_q", "计算维度": "(b,128,s,64)", "前驱算子": "linear(q_b)", "后继算子": "qkt_rope"},
                    {"阶段": "Prefill", "算子": "linear(kv_a)", "计算维度": "(b,s,7168)*(7168,576)", "前驱算子": "输入", "后继算子": "rope_k, rmsnorm(kv)"},
                    {"阶段": "Prefill", "算子": "rmsnorm(kv)", "计算维度": "(b,s,512)", "前驱算子": "linear(kv_a)", "后继算子": "kv_transpose"},
                    {"阶段": "Prefill", "算子": "rope_k", "计算维度": "(b,s,64)", "前驱算子": "linear(kv_a)", "后继算子": "qkt_rope"},
                    {"阶段": "Prefill", "算子": "qkt_rope", "计算维度": "(b,128,s,s)", "前驱算子": "rope_q, rope_k", "后继算子": "rope_add_nope"},
                    {"阶段": "Prefill", "算子": "q_absorb", "计算维度": "(b,128,s,128)*(128,512,512)", "前驱算子": "linear(q_b)", "后继算子": "q_absorb @ ctkv"},
                    {"阶段": "Prefill", "算子": "q_absorb @ ctkv", "计算维度": "(b,128,s,512)*(b,512,s)", "前驱算子": "q_absorb, kv_transpose", "后继算子": "rope_add_nope"},
                    {"阶段": "Prefill", "算子": "rope_add_nope", "计算维度": "(b,128,s,s)", "前驱算子": "qkt_rope, q_absorb @ ctkv", "后继算子": "softmax"},
                    {"阶段": "Prefill", "算子": "softmax", "计算维度": "(b,128,s,s)", "前驱算子": "rope_add_nope", "后继算子": "s_ctkv"},
                    {"阶段": "Prefill", "算子": "s_ctkv", "计算维度": "(b,128,s,s)*(b,s,512)", "前驱算子": "softmax", "后继算子": "sv_Wuv"},
                    {"阶段": "Prefill", "算子": "sv_Wuv", "计算维度": "(b,128,s,512)*(128,512,128)", "前驱算子": "s_ctkv", "后继算子": "linear_o"},
                    {"阶段": "Prefill", "算子": "linear_o", "计算维度": "(b,s,16384)*(16384,7168)", "前驱算子": "sv_Wuv", "后继算子": "mla_resadd"},
                    {"阶段": "Prefill", "算子": "mla_resadd", "计算维度": "(b,s,7168)", "前驱算子": "linear_o", "后继算子": "rmsnorm_moe"},
                    {"阶段": "Prefill", "算子": "rmsnorm_moe", "计算维度": "(b,s,7168)", "前驱算子": "mla_resadd", "后继算子": "router"},
                    {"阶段": "Prefill", "算子": "router", "计算维度": "(b,s,7168)*(7168,256)", "前驱算子": "rmsnorm_moe", "后继算子": "sigmoid"},
                    {"阶段": "Prefill", "算子": "sigmoid", "计算维度": "(b,s,256)", "前驱算子": "router", "后继算子": "rank_add"},
                    {"阶段": "Prefill", "算子": "rank_add", "计算维度": "(b,s,256)", "前驱算子": "sigmoid", "后继算子": "linear_up, linear_gate"},
                    {"阶段": "Prefill", "算子": "linear_up", "计算维度": "(b,s,7168)*(7168,2048)*9", "前驱算子": "rmsnorm_moe", "后继算子": "swiglu_mul"},
                    {"阶段": "Prefill", "算子": "linear_gate", "计算维度": "(b,s,7168)*(7168,2048)*9", "前驱算子": "rmsnorm_moe", "后继算子": "silu"},
                    {"阶段": "Prefill", "算子": "silu", "计算维度": "(b,s,2048)*9", "前驱算子": "linear_gate", "后继算子": "swiglu_mul"},
                    {"阶段": "Prefill", "算子": "swiglu_mul", "计算维度": "(b,s,2048)*9", "前驱算子": "linear_up, silu", "后继算子": "linear_down"},
                    {"阶段": "Prefill", "算子": "linear_down", "计算维度": "(b,s,2048)*(2048,7168)*9", "前驱算子": "swiglu_mul", "后继算子": "mul"},
                    {"阶段": "Prefill", "算子": "mul", "计算维度": "(b,s,7168)*9", "前驱算子": "linear_down", "后继算子": "moe_add"},
                    {"阶段": "Prefill", "算子": "moe_add", "计算维度": "(b,s,7168)*9", "前驱算子": "mul", "后继算子": "输出"},
                    
                    # Decode阶段
                    {"阶段": "Decode", "算子": "mla rmsnorm", "计算维度": "(b,1,7168)", "前驱算子": "输入", "后继算子": "linear(q_a)"},
                    {"阶段": "Decode", "算子": "linear(q_a)", "计算维度": "(b,1,7168)*(7168,1536)", "前驱算子": "mla rmsnorm", "后继算子": "rmsnorm(q)"},
                    {"阶段": "Decode", "算子": "rmsnorm(q)", "计算维度": "(b,1,1536)", "前驱算子": "linear(q_a)", "后继算子": "linear(q_b)"},
                    {"阶段": "Decode", "算子": "linear(q_b)", "计算维度": "(b,1,1536)*(1536,16384)", "前驱算子": "rmsnorm(q)", "后继算子": "rope_q"},
                    {"阶段": "Decode", "算子": "rope_q", "计算维度": "(b,128,1,64)", "前驱算子": "linear(q_b)", "后继算子": "qkt_rope"},
                    {"阶段": "Decode", "算子": "linear(kv_a)", "计算维度": "(b,1,7168)*(7168,576)", "前驱算子": "输入", "后继算子": "rope_k, rmsnorm(kv)"},
                    {"阶段": "Decode", "算子": "rmsnorm(kv)", "计算维度": "(b,1,512)", "前驱算子": "linear(kv_a)", "后继算子": "kv_transpose"},
                    {"阶段": "Decode", "算子": "rope_k", "计算维度": "(b,1,64)", "前驱算子": "linear(kv_a)", "后继算子": "qkt_rope"},
                    {"阶段": "Decode", "算子": "qkt_rope", "计算维度": "(b,128,1,context_lenth+1)", "前驱算子": "rope_q, rope_k", "后继算子": "rope_add_nope"},
                    {"阶段": "Decode", "算子": "q_absorb", "计算维度": "(b,128,1,128)*(128,512,512)", "前驱算子": "linear(q_b)", "后继算子": "q_absorb @ ctkv"},
                    {"阶段": "Decode", "算子": "q_absorb @ ctkv", "计算维度": "(b,128,1,512)*(b,512,context_lenth+1)", "前驱算子": "q_absorb, kv_transpose", "后继算子": "rope_add_nope"},
                    {"阶段": "Decode", "算子": "rope_add_nope", "计算维度": "(b,128,1,context_lenth+1)", "前驱算子": "qkt_rope, q_absorb @ ctkv", "后继算子": "softmax"},
                    {"阶段": "Decode", "算子": "softmax", "计算维度": "(b,128,1,context_lenth+1)", "前驱算子": "rope_add_nope", "后继算子": "s_ctkv"},
                    {"阶段": "Decode", "算子": "s_ctkv", "计算维度": "(b,128,1,context_lenth+1)*(b,context_lenth+1,512)", "前驱算子": "softmax", "后继算子": "sv_Wuv"},
                    {"阶段": "Decode", "算子": "sv_Wuv", "计算维度": "(b,128,1,512)*(128,512,128)", "前驱算子": "s_ctkv", "后继算子": "linear_o"},
                    {"阶段": "Decode", "算子": "linear_o", "计算维度": "(b,1,16384)*(16384,7168)", "前驱算子": "sv_Wuv", "后继算子": "mla_resadd"},
                    {"阶段": "Decode", "算子": "mla_resadd", "计算维度": "(b,1,7168)", "前驱算子": "linear_o", "后继算子": "rmsnorm_moe"},
                    {"阶段": "Decode", "算子": "rmsnorm_moe", "计算维度": "(b,1,7168)", "前驱算子": "mla_resadd", "后继算子": "router"},
                    {"阶段": "Decode", "算子": "router", "计算维度": "(b,1,7168)*(7168,256)", "前驱算子": "rmsnorm_moe", "后继算子": "sigmoid"},
                    {"阶段": "Decode", "算子": "sigmoid", "计算维度": "(b,1,256)", "前驱算子": "router", "后继算子": "rank_add"},
                    {"阶段": "Decode", "算子": "rank_add", "计算维度": "(b,1,256)", "前驱算子": "sigmoid", "后继算子": "linear_up, linear_gate"},
                    {"阶段": "Decode", "算子": "linear_up", "计算维度": "(b,1,7168)*(7168,2048)*9", "前驱算子": "rmsnorm_moe", "后继算子": "swiglu_mul"},
                    {"阶段": "Decode", "算子": "linear_gate", "计算维度": "(b,1,7168)*(7168,2048)*9", "前驱算子": "rmsnorm_moe", "后继算子": "silu"},
                    {"阶段": "Decode", "算子": "silu", "计算维度": "(b,1,2048)*9", "前驱算子": "linear_gate", "后继算子": "swiglu_mul"},
                    {"阶段": "Decode", "算子": "swiglu_mul", "计算维度": "(b,1,2048)*9", "前驱算子": "linear_up, silu", "后继算子": "linear_down"},
                    {"阶段": "Decode", "算子": "linear_down", "计算维度": "(b,1,2048)*(2048,7168)*9", "前驱算子": "swiglu_mul", "后继算子": "mul"},
                    {"阶段": "Decode", "算子": "mul", "计算维度": "(b,1,7168)*9", "前驱算子": "linear_down", "后继算子": "moe_add"},
                    {"阶段": "Decode", "算子": "moe_add", "计算维度": "(b,1,7168)*9", "前驱算子": "mul", "后继算子": "输出"}
                ]
            elif model_name == "seed_oss":
                compute_graph_data = [
                    # Prefill阶段
                    {"阶段": "Prefill", "算子": "attn_rmsnorm", "计算维度": "(b,s,5120)", "前驱算子": "输入", "后继算子": "Q_proj"},
                    {"阶段": "Prefill", "算子": "Q_proj", "计算维度": "(b,s,5120)*(5120,80*128)", "前驱算子": "attn_rmsnorm", "后继算子": "Q_rope"},
                    {"阶段": "Prefill", "算子": "Q_rope", "计算维度": "(b,80,s,128)", "前驱算子": "Q_proj", "后继算子": "QKT"},
                    {"阶段": "Prefill", "算子": "K_proj", "计算维度": "(b,s,5120)*(5120,8*128)", "前驱算子": "attn_rmsnorm", "后继算子": "K_rope"},
                    {"阶段": "Prefill", "算子": "K_rope", "计算维度": "(b,8,s,128)", "前驱算子": "K_proj", "后继算子": "QKT"},
                    {"阶段": "Prefill", "算子": "V_proj", "计算维度": "(b,s,5120)*(5120,8*128)", "前驱算子": "attn_rmsnorm", "后继算子": "SV"},
                    {"阶段": "Prefill", "算子": "QKT", "计算维度": "(b,80,s,128)*(b,8,s,128)", "前驱算子": "Q_rope, K_rope", "后继算子": "softmax"},
                    {"阶段": "Prefill", "算子": "softmax", "计算维度": "(b,80,s,s)", "前驱算子": "QKT", "后继算子": "SV"},
                    {"阶段": "Prefill", "算子": "SV", "计算维度": "(b,80,s,s)*(b,8,s,128)", "前驱算子": "softmax, V_proj", "后继算子": "O_proj"},
                    {"阶段": "Prefill", "算子": "O_proj", "计算维度": "(b,s,80*128)*(80*128,5120)", "前驱算子": "SV", "后继算子": "attn_resadd"},
                    {"阶段": "Prefill", "算子": "attn_resadd", "计算维度": "(b,s,5120)", "前驱算子": "O_proj", "后继算子": "ffn_rmsnorm"},
                    {"阶段": "Prefill", "算子": "ffn_rmsnorm", "计算维度": "(b,s,5120)", "前驱算子": "attn_resadd", "后继算子": "linear_up"},
                    {"阶段": "Prefill", "算子": "linear_up", "计算维度": "(b,s,5120)*(5120,27648)", "前驱算子": "ffn_rmsnorm", "后继算子": "swiglu_mul"},
                    {"阶段": "Prefill", "算子": "linear_gate", "计算维度": "(b,s,5120)*(5120,27648)", "前驱算子": "ffn_rmsnorm", "后继算子": "silu"},
                    {"阶段": "Prefill", "算子": "silu", "计算维度": "(b,s,27648)", "前驱算子": "linear_gate", "后继算子": "swiglu_mul"},
                    {"阶段": "Prefill", "算子": "swiglu_mul", "计算维度": "(b,s,27648)", "前驱算子": "linear_up, silu", "后继算子": "linear_down"},
                    {"阶段": "Prefill", "算子": "linear_down", "计算维度": "(b,s,27648)*(27648,5120)", "前驱算子": "swiglu_mul", "后继算子": "ffn_resadd"},
                    {"阶段": "Prefill", "算子": "ffn_resadd", "计算维度": "(b,s,5120)", "前驱算子": "linear_down", "后继算子": "输出"},
                    
                    # Decode阶段
                    {"阶段": "Decode", "算子": "attn_rmsnorm", "计算维度": "(b,1,5120)", "前驱算子": "输入", "后继算子": "Q_proj"},
                    {"阶段": "Decode", "算子": "Q_proj", "计算维度": "(b,1,5120)*(5120,80*128)", "前驱算子": "attn_rmsnorm", "后继算子": "Q_rope"},
                    {"阶段": "Decode", "算子": "Q_rope", "计算维度": "(b,80,1,128)", "前驱算子": "Q_proj", "后继算子": "QKT"},
                    {"阶段": "Decode", "算子": "K_proj", "计算维度": "(b,1,5120)*(5120,8*128)", "前驱算子": "attn_rmsnorm", "后继算子": "K_rope"},
                    {"阶段": "Decode", "算子": "K_rope", "计算维度": "(b,8,1,128)", "前驱算子": "K_proj", "后继算子": "QKT"},
                    {"阶段": "Decode", "算子": "V_proj", "计算维度": "(b,1,5120)*(5120,8*128)", "前驱算子": "attn_rmsnorm", "后继算子": "SV"},
                    {"阶段": "Decode", "算子": "QKT", "计算维度": "(b,80,1,128)*(b,8,128,context_lenth+1)", "前驱算子": "Q_rope, K_rope", "后继算子": "softmax"},
                    {"阶段": "Decode", "算子": "softmax", "计算维度": "(b,80,1,context_lenth+1)", "前驱算子": "QKT", "后继算子": "SV"},
                    {"阶段": "Decode", "算子": "SV", "计算维度": "(b,80,1,context_lenth+1)*(b,80,context_lenth+1,128)", "前驱算子": "softmax, V_proj", "后继算子": "O_proj"},
                    {"阶段": "Decode", "算子": "O_proj", "计算维度": "(b,1,80*128)*(80*128,5120)", "前驱算子": "SV", "后继算子": "attn_resadd"},
                    {"阶段": "Decode", "算子": "attn_resadd", "计算维度": "(b,1,5120)", "前驱算子": "O_proj", "后继算子": "ffn_rmsnorm"},
                    {"阶段": "Decode", "算子": "ffn_rmsnorm", "计算维度": "(b,1,5120)", "前驱算子": "attn_resadd", "后继算子": "linear_up"},
                    {"阶段": "Decode", "算子": "linear_up", "计算维度": "(b,1,5120)*(5120,27648)", "前驱算子": "ffn_rmsnorm", "后继算子": "swiglu_mul"},
                    {"阶段": "Decode", "算子": "linear_gate", "计算维度": "(b,1,5120)*(5120,27648)", "前驱算子": "ffn_rmsnorm", "后继算子": "silu"},
                    {"阶段": "Decode", "算子": "silu", "计算维度": "(b,1,27648)", "前驱算子": "linear_gate", "后继算子": "swiglu_mul"},
                    {"阶段": "Decode", "算子": "swiglu_mul", "计算维度": "(b,1,27648)", "前驱算子": "linear_up, silu", "后继算子": "linear_down"},
                    {"阶段": "Decode", "算子": "linear_down", "计算维度": "(b,1,27648)*(27648,5120)", "前驱算子": "swiglu_mul", "后继算子": "ffn_resadd"},
                    {"阶段": "Decode", "算子": "ffn_resadd", "计算维度": "(b,1,5120)", "前驱算子": "linear_down", "后继算子": "输出"}
                ]

            compute_graph_df = pd.DataFrame(compute_graph_data)

            # 收集模型数据
            model_data = collect_model_data(
                model_name,
                prefill_class,
                decode_class,
                hidden_size,
                datatype,
                device,
                batch_size,
                prefill_lenth,
                decode_lenth
            )

            # 查找空字典的位置
            empty_indices = [i for i, item in enumerate(model_data) if item == {}]

            # 提取prefill_operators (从开始到第一个空字典之前的部分，排除汇总行)
            prefill_operators = []
            for i in range(empty_indices[0]):
                if model_data[i].get('算子') != '汇总':
                    prefill_operators.append(model_data[i])

            # 提取prefill_summary (prefill部分的汇总行)
            prefill_summary = []
            for i in range(empty_indices[0]):
                if model_data[i].get('算子') == '汇总':
                    prefill_summary.append(model_data[i])

            # 提取decode_operators (从第二个空字典之后到末尾的部分，排除汇总行)
            decode_operators = []
            for i in range(empty_indices[1] + 1, len(model_data)):
                if model_data[i].get('算子') != '汇总':
                    decode_operators.append(model_data[i])

            # 提取decode_summary (decode部分的汇总行)
            decode_summary = []
            for i in range(empty_indices[1] + 1, len(model_data)):
                if model_data[i].get('算子') == '汇总':
                    decode_summary.append(model_data[i])

            # 合并operators和summaries
            operators_data = prefill_operators +[{}] +[{}] + decode_operators
            summary_data = prefill_summary + decode_summary

            # 创建DataFrames
            operators_df = pd.DataFrame(operators_data)
            summary_df = pd.DataFrame(summary_data)
            model_df = pd.DataFrame(model_data)  # 保留原有的性能数据

            # 询问保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="保存性能数据"
            )

            if file_path:
                # 导出到Excel
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    # 创建靠左对齐的格式
                    left_align_format = writer.book.add_format({'align': 'left'})

                    # 算子数据工作表
                    operators_df.to_excel(writer, sheet_name='算子数据', index=False)
                    worksheet = writer.sheets['算子数据']
                    # 设置列宽和对齐方式
                    worksheet.set_column('A:A', 10, left_align_format)  # 调整第一列宽度并靠左对齐
                    worksheet.set_column('B:B', 10, left_align_format)  # 调整第二列宽度并靠左对齐
                    worksheet.set_column('C:C', 15, left_align_format)  # 调整第三列宽度并靠左对齐
                    worksheet.set_column('D:E', 20, left_align_format)

                    # 汇总数据工作表
                    summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
                    worksheet = writer.sheets['汇总数据']
                    # 设置列宽和对齐方式
                    worksheet.set_column('A:A', 10, left_align_format)  # 调整第一列宽度并靠左对齐
                    worksheet.set_column('B:B', 10, left_align_format)  # 调整第二列宽度并靠左对齐
                    worksheet.set_column('C:C', 15, left_align_format)  # 调整第三列宽度并靠左对齐
                    worksheet.set_column('D:G', 15, left_align_format)
                    worksheet.set_column('H:H', 20, left_align_format)
                    worksheet.set_column('I:K', 20, left_align_format)

                    # 计算图数据工作表
                    compute_graph_df.to_excel(writer, sheet_name='计算图数据', index=False)
                    worksheet = writer.sheets['计算图数据']
                    # 设置列宽和对齐方式
                    worksheet.set_column('A:A', 10, left_align_format)  # 阶段列宽度并靠左对齐
                    worksheet.set_column('B:B', 20, left_align_format)  # 算子列宽度并靠左对齐
                    worksheet.set_column('C:C', 35, left_align_format)  # 计算维度列宽度并靠左对齐
                    worksheet.set_column('D:D', 25, left_align_format)  # 前驱算子列宽度并靠左对齐
                    worksheet.set_column('E:E', 25, left_align_format)  # 后继算子列宽度并靠左对齐

                    # 硬件参数工作表
                    hardware_df = pd.DataFrame(list(self.device_info.items()), columns=['参数', '值'])
                    hardware_df.to_excel(writer, sheet_name='硬件参数', index=False)
                    worksheet = writer.sheets['硬件参数']
                    worksheet.set_column('A:A', 30, left_align_format)  # 参数列宽度并靠左对齐
                    worksheet.set_column('B:B', 15, left_align_format)  # 值列宽度并靠左对齐

                    # 模型结构工作表
                    model_df = pd.DataFrame(list(self.model_info.items()), columns=['参数', '值'])
                    model_df.to_excel(writer, sheet_name='模型结构', index=False)
                    worksheet = writer.sheets['模型结构']
                    worksheet.set_column('A:A', 20, left_align_format)  # 参数列宽度并靠左对齐
                    worksheet.set_column('B:B', 15, left_align_format)  # 值列宽度并靠左对齐

                messagebox.showinfo("成功", f"数据已成功导出到: {file_path}")
            else:
                messagebox.showinfo("取消", "导出操作已取消")

            # 关闭加载窗口
            loading_window.destroy()
            # 启用运行按钮
            self.run_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("错误", f"计算过程中出错: {str(e)}")
            self.run_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = PerformanceModelGUI(root)
    root.mainloop()