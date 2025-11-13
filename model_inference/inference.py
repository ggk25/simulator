import sys
import os
import json
import pandas as pd
import numpy as np
import math
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个模块的类
from software_model.GQA import Prefill as GQAPrefill, Decode as GQADecode
from software_model.MoE import Prefill as MoEPrefill, Decode as MoEDecode
from software_model.FFN import Prefill as FFNPrefill, Decode as FFNDecode
from software_model.MLA import Prefill as MLAPrefill, Decode as MLADecode
# 旧有的预定义模型导入已移除，统一使用灵活配置的模型
from utils import DataType, Tensor, data_type_dict
from hardware_model.device import device_dict
from power import get_global_power_counter
from power.energy_table import load_energy_table
from model_inference.compute_power import compute_power_from_inference_data

class ModelConfig:
    """模型配置类，用于定义模型的结构"""
    def __init__(self, name, attention_type="GQA", ffn_type="FFN", **kwargs):
        self.name = name
        self.attention_type = attention_type  # "GQA" 或 "MLA"
        self.ffn_type = ffn_type  # "FFN" 或 "MoE"
        
        # 通用参数
        self.hidden_size = kwargs.get('hidden_size', 5120)
        self.datatype = kwargs.get('datatype', data_type_dict["fp8"])
        
        # GQA参数
        self.head_dim = kwargs.get('head_dim', 128)
        self.num_attention_heads = kwargs.get('num_attention_heads', 80)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 8)
        self.intermediate_size = kwargs.get('intermediate_size', 27648)
        
        # MLA参数
        self.q_compress_dim = kwargs.get('q_compress_dim', 1536)
        self.qk_rope_dim = kwargs.get('qk_rope_dim', 64)
        self.kv_compress_dim = kwargs.get('kv_compress_dim', 576)
        self.n_heads = kwargs.get('n_heads', 128)
        self.qkv_dim = kwargs.get('qkv_dim', 128)
        
        # MoE参数
        self.experts_dim = kwargs.get('experts_dim', 2048)
        self.shared_experts_count = kwargs.get('shared_experts_count', 1)
        self.selected_expert_count = kwargs.get('selected_expert_count', 8)
        self.experts_count = kwargs.get('experts_count', 256)
        
        # FFN参数
        self.ffn_intermediate_size = kwargs.get('ffn_intermediate_size', 27648)

        self.layer_count = kwargs.get('layer_count', 64)


class FlexibleModel:
    """灵活的模型类，根据配置组合不同的模块"""
    def __init__(self, config, stage="prefill", context_length=None):
        self.config = config
        self.stage = stage
        self.context_length = context_length
        
        # 根据配置选择attention模块
        if config.attention_type == "GQA":
            if stage == "prefill":
                self.attention = GQAPrefill(
                    datatype=config.datatype,
                    hidden_size=config.hidden_size,
                    head_dim=config.head_dim,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads
                )
            else:  # decode
                self.attention = GQADecode(
                    datatype=config.datatype,
                    context_lenth=context_length,
                    hidden_size=config.hidden_size,
                    head_dim=config.head_dim,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                )
        elif config.attention_type == "MLA":
            if stage == "prefill":
                self.attention = MLAPrefill(
                    datatype=config.datatype,
                    hiden_states=config.hidden_size,
                    q_compress_dim=config.q_compress_dim,
                    qk_rope_dim=config.qk_rope_dim,
                    kv_compress_dim=config.kv_compress_dim,
                    n_heads=config.n_heads,
                    qkv_dim=config.qkv_dim
                )
            else:  # decode
                self.attention = MLADecode(
                    datatype=config.datatype,
                    context_lenth=context_length,
                    hiden_states=config.hidden_size,
                    q_compress_dim=config.q_compress_dim,
                    qk_rope_dim=config.qk_rope_dim,
                    kv_compress_dim=config.kv_compress_dim,
                    n_heads=config.n_heads,
                    qkv_dim=config.qkv_dim
                )
        
        # 根据配置选择FFN模块
        if config.ffn_type == "FFN":
            if stage == "prefill":
                self.ffn = FFNPrefill(
                    datatype=config.datatype,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_intermediate_size
                )
                # 设置hidden_size，因为FFN模块需要这个属性
                self.ffn.hidden_size = config.hidden_size
            else:  # decode
                self.ffn = FFNDecode(
                    datatype=config.datatype,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_intermediate_size
                )
                # 设置hidden_size，因为FFN模块需要这个属性
                self.ffn.hidden_size = config.hidden_size
        elif config.ffn_type == "MoE":
            if stage == "prefill":
                self.ffn = MoEPrefill(
                    datatype=config.datatype,
                    hiden_states=config.hidden_size,
                    experts_dim=config.experts_dim,
                    shared_experts_count=config.shared_experts_count,
                    selected_expert_count=config.selected_expert_count,
                    experts_count=config.experts_count
                )
            else:  # decode
                self.ffn = MoEDecode(
                    datatype=config.datatype,
                    context_lenth=context_length,
                    hiden_states=config.hidden_size,
                    experts_dim=config.experts_dim,
                    shared_experts_count=config.shared_experts_count,
                    selected_expert_count=config.selected_expert_count,
                    experts_count=config.experts_count
                )
    
    def __call__(self, input_tensor):
        # 先执行attention
        attention_output = self.attention(input_tensor)
        # 再执行FFN
        output = self.ffn(attention_output)
        return output
    
    def mapping_and_simulate(self, device):
        # 分别获取attention和FFN的延迟数据
        attn_operator_latency, attn_total_latency = self.attention.mapping_and_simulate(device)
        
        # MoE模块返回三个值
        ffn_operator_latency, ffn_total_latency, pipeline_latency = self.ffn.mapping_and_simulate(device)
        
        # 合并算子延迟
        all_operator_latency = attn_operator_latency + ffn_operator_latency
        total_latency = attn_total_latency + ffn_total_latency
        
        return all_operator_latency, total_latency, pipeline_latency


def collect_model_data(model_config, device, batch_size, prefill_lenth, decode_lenth, use_flexible_model=True):
    """
    收集指定模型的延迟数据
    """
    micro_batch = batch_size / device.n_chip
    
    # 统一走灵活模型路径（自定义 JSON 配置）
    model_name = model_config.name
    hidden_size = model_config.hidden_size
    datatype = model_config.datatype
    energy_table_local = load_energy_table()
    sram_leakage_w_total = float(getattr(energy_table_local, 'sram_leakage_power', 0.0))

    # 收集prefill阶段数据
    prompt = Tensor([micro_batch, prefill_lenth, hidden_size], data_type=datatype)
    prefill_model = FlexibleModel(model_config, stage="prefill")
    prefill_output = prefill_model(prompt)
    prefill_operator_latency, prefill_latency, pipeline_latency = prefill_model.mapping_and_simulate(device)
    # 收集prefill阶段算子能耗（来自各模块的 operator_energy）
    prefill_operator_energy = []
    prefill_operator_energy.extend(getattr(prefill_model.attention, 'operator_energy', []) or [])
    prefill_operator_energy.extend(getattr(prefill_model.ffn, 'operator_energy', []) or [])
    
    # 计算TTFT (Time To First Token)
    prefill_latency_ms = (prefill_latency * math.ceil(model_config.layer_count/device.n_chip) + pipeline_latency) / (device.frequency * 1e6) * 1e3 #单芯片prefill某几层的延时
    ttft = prefill_latency_ms * device.n_chip

    # 处理prefill阶段数据,单层在单芯片上的延迟数据
    prefill_data = []
    total_prefill_compute = sum(op['计算延时'] for op in prefill_operator_latency)
    total_prefill_comm = sum(op['通信延时'] for op in prefill_operator_latency)
    total_prefill = total_prefill_compute + total_prefill_comm
    
    # 转换为ms单位，乘device.n_chip是因为总共device.n_chip个芯片
    total_prefill_compute_ms = (total_prefill_compute / (device.frequency * 1e6) * 1e3)* math.ceil(model_config.layer_count/device.n_chip)*device.n_chip
    total_prefill_comm_ms = (total_prefill_comm* math.ceil(model_config.layer_count/device.n_chip) + pipeline_latency) / (device.frequency * 1e6) * 1e3*device.n_chip
    total_prefill_ms = (total_prefill* math.ceil(model_config.layer_count/device.n_chip) + pipeline_latency) / (device.frequency * 1e6) * 1e3*device.n_chip
    
    for op in prefill_operator_latency:
        op_name = op['operator']
        total = op['总延时']
        
        # 转换为ms单位
        total_ms = total / (device.frequency * 1e6) * 1e3
        
        # 计算算子占总延时的比例
        op_ratio = total / total_prefill if total_prefill > 0 else 0
        
        prefill_data.append({
            '模型': model_name,
            '阶段': 'prefill',
            '算子': op_name,
            '总延时(ms)': total_ms,
            '算子占总延时比例': op_ratio
        })

    # 生成prefill阶段能耗数据
    prefill_energy_rows = []
    prefill_energy_scale = 1
    for e in prefill_operator_energy:
        op_name = e.get('operator')
        total = float(e.get('总延时', 0.0))
        energy_pj = float(e.get('能耗', 0.0)) * prefill_energy_scale
        total_ms = total / (device.frequency * 1e6) * 1e3 
        avg_power_w = (energy_pj / total_ms * 1e-9 + sram_leakage_w_total) if total_ms > 0 else None
        prefill_energy_rows.append({
            '模型': model_name,
            '阶段': 'prefill',
            '算子': op_name,
            '总延时(ms)': total_ms,
            '能耗(pJ)': energy_pj,
            '平均功耗(W)': avg_power_w
        })
    
    # 添加prefill阶段汇总行
    prefill_data.append({
        '模型': model_name,
        '阶段': 'prefill',
        '算子': '汇总',
        '计算延时(ms)': total_prefill_compute_ms,
        '通信延时(ms)': total_prefill_comm_ms,
        '总延时(ms)': total_prefill_ms,
        '计算延时占比': total_prefill_compute / total_prefill if total_prefill > 0 else 0,
        '通信延时占比': total_prefill_comm / total_prefill if total_prefill > 0 else 0,
        'TTFT(s)': ttft/1000
    })
    
    # 收集decode阶段数据
    decode_data = []
    decode_energy_rows = []
    # 用于累积每个算子的延时
    op_total = {}
    # 用于累积每个算子的能耗
    op_energy_total = {}
    
    total_decode_compute = 0
    total_decode_comm = 0
    total_decode = 0
    total_pipeline_latency = 0
    
    # 收集所有decode步骤的数据 (1到decode_lenth)
    for i in range(1, decode_lenth + 1):
        context_lenth = prefill_lenth + i
        decode_model = FlexibleModel(model_config, stage="decode", context_length=context_lenth)
        input = Tensor([micro_batch, 1, hidden_size], data_type=datatype)
        output = decode_model(input)
        decode_operator_latency, decode_latency, pipeline_latency = decode_model.mapping_and_simulate(device)
        # 取该步的能耗明细
        step_operator_energy = []
        step_operator_energy.extend(getattr(decode_model.attention, 'operator_energy', []) or [])
        step_operator_energy.extend(getattr(decode_model.ffn, 'operator_energy', []) or [])
        
        # 累加当前步骤的延时
        step_compute = sum(op['计算延时'] for op in decode_operator_latency)
        step_comm = sum(op['通信延时'] for op in decode_operator_latency)
        step_total = step_compute + step_comm
        
        total_decode_compute += step_compute
        total_decode_comm += step_comm
        total_pipeline_latency += pipeline_latency
        total_decode += step_total
        
        # 为该步写入每个算子的延时行
        for op in decode_operator_latency:
            op_name = op['operator']
            total = op['总延时']
            total_ms = total / (device.frequency * 1e6) * 1e3
            op_ratio = total / step_total if step_total > 0 else 0
            decode_data.append({
                '模型': model_name,
                '阶段': 'decode',
                'decode步长': i,
                '算子': op_name,
                '总延时(ms)': total_ms,
                '算子占总延时比例': op_ratio
            })

        # 为该步写入每个算子的能耗行
        for e in step_operator_energy:
            op_name = e.get('operator')
            total = float(e.get('总延时', 0.0))
            energy_pj = float(e.get('能耗', 0.0))
            total_ms = total / (device.frequency * 1e6) * 1e3
            avg_power_w = (energy_pj / total_ms * 1e-9 + sram_leakage_w_total) if total_ms > 0 else None
            decode_energy_rows.append({
                '模型': model_name,
                '阶段': 'decode',
                'decode步长': i,
                '算子': op_name,
                '总延时(ms)': total_ms,
                '能耗(pJ)': energy_pj,
                '平均功耗(W)': avg_power_w
            })

        # 累加每个算子的延时（用于阶段汇总）
        for op in decode_operator_latency:
            op_name = op['operator']
            total = op['总延时']
            
            if op_name not in op_total:
                op_total[op_name] = 0
            
            op_total[op_name] += total
        # 累加每个算子的能耗（用于阶段汇总）
        for e in step_operator_energy:
            op_name = e.get('operator')
            op_energy_total[op_name] = op_energy_total.get(op_name, 0.0) + float(e.get('能耗', 0.0))
    
    # 转换为ms单位
    total_decode_compute_ms = total_decode_compute * math.ceil(model_config.layer_count/device.n_chip) / (device.frequency * 1e6) * 1e3*device.n_chip
    total_decode_comm_ms = (total_decode_comm * math.ceil(model_config.layer_count/device.n_chip) + total_pipeline_latency) / (device.frequency * 1e6) * 1e3*device.n_chip
    total_decode_ms = (total_decode* math.ceil(model_config.layer_count/device.n_chip) + total_pipeline_latency) / (device.frequency * 1e6) * 1e3*device.n_chip
    
    # 计算端对端吞吐率
    throughput = batch_size * decode_lenth * 1e3 / (total_decode_ms + total_prefill_ms)
    TBT = total_decode_ms / decode_lenth
    
    # 处理decode阶段每个算子的累积数据（阶段级总计）
    for op_name in op_total:
        total = op_total[op_name]
        
        # 转换为ms单位
        total_ms = total / (device.frequency * 1e6) * 1e3* math.ceil(model_config.layer_count/device.n_chip)*device.n_chip
        
        # 计算算子占总延时的比例
        op_ratio = total / total_decode if total_decode > 0 else 0
        
        decode_data.append({
            '模型': model_name,
            '阶段': 'decode',
            '算子': op_name,
            '总延时(ms)': total_ms,
            '算子占总延时比例': op_ratio
        })
        # 能耗（按层数×芯片数缩放，与延时一致）
        energy_scale = math.ceil(model_config.layer_count / device.n_chip) * device.n_chip
        op_energy_scaled = op_energy_total.get(op_name, 0.0) * energy_scale
        avg_power_w = (op_energy_scaled / total_ms * 1e-9 + sram_leakage_w_total) if total_ms > 0 else None
        decode_energy_rows.append({
            '模型': model_name,
            '阶段': 'decode',
            '算子': op_name,
            '总延时(ms)': total_ms,
            '能耗(pJ)': op_energy_scaled,
            '平均功耗(W)': avg_power_w
        })
    
    # 添加decode阶段汇总行
    decode_data.append({
        '模型': model_name,
        '阶段': 'decode',
        '算子': '汇总',
        '计算延时(ms)': total_decode_compute_ms,
        '通信延时(ms)': total_decode_comm_ms,
        '总延时(ms)': total_decode_ms,
        '计算延时占比': total_decode_compute / total_decode if total_decode > 0 else 0,
        '通信延时占比': total_decode_comm / total_decode if total_decode > 0 else 0,
        '吞吐率(tokens/s)': throughput,
        '生成速度(tokens/s)':1000/TBT
    })
    
    # 合并数据，将prefill和decode的汇总行分别放在各自部分的最后
    # 先处理prefill数据，将汇总行放在最后
    prefill_operators = [item for item in prefill_data if item.get('算子') != '汇总']
    prefill_summary = [item for item in prefill_data if item.get('算子') == '汇总']
    
    # 再处理decode数据，将汇总行放在最后
    decode_operators = [item for item in decode_data if item.get('算子') != '汇总']
    decode_summary = [item for item in decode_data if item.get('算子') == '汇总']
    
    # 返回合并后的数据，汇总行分别放在prefill和decode部分的最后
    # 组合能耗sheet数据（prefill + decode）
    operator_energy_rows = prefill_energy_rows + [{'模型': model_name, '阶段': 'prefill', '算子': '汇总'}] + decode_energy_rows + [{'模型': model_name, '阶段': 'decode', '算子': '汇总'}]

    return prefill_operators + prefill_summary + [{}] + [{}] + decode_operators + decode_summary, operator_energy_rows


def _load_json_configs(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 支持单模型或列表
    if isinstance(data, dict):
        data = [data]
    return data

def _build_model_config(entry):
    # 运行参数(非模型结构)提取并返回 (batch_size, prefill_lenth, decode_lenth, device_name)
    runtime = {
        'batch_size': entry.pop('batch_size', 256),
        'prefill_lenth': entry.pop('prefill_length', entry.pop('prefill_lenth', 1024)),
        'decode_lenth': entry.pop('decode_length', entry.pop('decode_lenth', 2048)),
        'device_name': entry.pop('device', 'D37x')
    }
    # datatype 字符串转换
    dt = entry.get('datatype', 'fp8')
    entry['datatype'] = data_type_dict.get(dt, data_type_dict['fp8'])
    model_cfg = ModelConfig(**entry)
    return model_cfg, runtime

def main():
    # JSON 配置文件路径（位于当前目录）
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    try:
        raw_entries = _load_json_configs(config_path)
    except Exception as e:
        print(f"加载配置失败: {e}\n使用内置默认模型。可创建 {config_path} 进行自定义。")
        raw_entries = [{
            'name': 'seedoss', 'attention_type': 'GQA', 'ffn_type': 'FFN', 'hidden_size': 5120,
            'head_dim':128,'num_attention_heads':80,'num_key_value_heads':8,'ffn_intermediate_size':27648,
            'layer_count':64,'datatype':'fp8','batch_size':256,'prefill_length':1024,'decode_length':2048,
            'device':'D37x'
        }]

    # 收集所有模型的数据
    all_data = []
    summary_data = []

    # 复制以避免修改原数据
    entry_copy = dict(raw_entries[0])
    model_cfg, runtime = _build_model_config(entry_copy)
    device = device_dict.get(runtime['device_name'], device_dict['D37x'])
    batch_size = runtime['batch_size']
    prefill_lenth = runtime['prefill_lenth']
    decode_lenth = runtime['decode_lenth']
    print(f"[配置] 模型={model_cfg.name} device={runtime['device_name']} batch={batch_size} prefill={prefill_lenth} decode={decode_lenth}")

    model_data, energy_op_rows = collect_model_data(
        model_cfg,
        device=device,
        batch_size=batch_size,
        prefill_lenth=prefill_lenth,
        decode_lenth=decode_lenth,
        use_flexible_model=True
    )
    all_data.extend(model_data)
    energy_all_rows = []
    energy_all_rows.extend(energy_op_rows)

    prefill_summary = None
    decode_summary = None
    for item in model_data:
        if item.get('算子') == '汇总' and item.get('阶段') == 'prefill':
            prefill_summary = item
        elif item.get('算子') == '汇总' and item.get('阶段') == 'decode':
            decode_summary = item
    if prefill_summary and decode_summary:
        summary_data.append({
            '模型': model_cfg.name,
            '首词延迟(TTFT)(s)': prefill_summary.get('TTFT(s)', 0),
            '总延时(ms)': prefill_summary.get('总延时(ms)', 0) + decode_summary.get('总延时(ms)', 0),
            '吞吐率(tokens/s)': decode_summary.get('吞吐率(tokens/s)', 0),
            '生成速度(tokens/s)': decode_summary.get('生成速度(tokens/s)', 0),
            'Prefill计算延时(ms)': prefill_summary.get('计算延时(ms)', 0),
            'Prefill通信延时(ms)': prefill_summary.get('通信延时(ms)', 0),
            'Decode计算延时(ms)': decode_summary.get('计算延时(ms)', 0),
            'Decode通信延时(ms)': decode_summary.get('通信延时(ms)', 0)
        })
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    energy_op_df = pd.DataFrame(energy_all_rows)
    summary_df = pd.DataFrame(summary_data)
    # 能耗统计
    energy_table = load_energy_table()
    power_counter = get_global_power_counter()
    energy_results = power_counter.compute_energy(energy_table)
    # 能耗按层数×pipeline级数进行缩放（与延时一致）
    # 根据输入包含的模型层数与芯片数估算缩放
    try:
        # 从汇总数据推导层数（取第一个模型配置的层数），如不可得则回退为1
        first_layers = None
        if isinstance(raw_entries, list) and len(raw_entries) > 0:
            first_layers = raw_entries[0].get('layer_count', None) if isinstance(raw_entries[0], dict) else None
        if first_layers is None and '层数' in df.columns:
            first_layers = int(df['层数'].iloc[0])
        if first_layers is None:
            first_layers = 1
        energy_scale = math.ceil(first_layers / device.n_chip) * device.n_chip
    except Exception:
        energy_scale = 1
    # 计算各算子累计时长（ms），用于算子平均功耗
    try:
        op_time_ms = (
            df[(df.get('算子').notna()) & (df['算子'] != '汇总')][['算子', '总延时(ms)']]
            .groupby('算子', as_index=True)
            .sum()
            .to_dict()['总延时(ms)']
        )
    except Exception:
        op_time_ms = {}
    # 生成能耗DataFrame（去掉总计专用键）
    energy_rows = []
    for op, vals in energy_results.items():
        if op == "__total__":
            continue
        row = {"算子": op}
        row.update({
            "MAC能耗(pJ)": vals["mac_energy_pj"] * energy_scale,
            "逐元素能耗(pJ)": vals["eltwise_energy_pj"] * energy_scale,
            "NoC能耗(pJ)": vals["noc_energy_pj"] * energy_scale,
            "PCIe能耗(pJ)": vals["pcie_energy_pj"] * energy_scale,
            "DRAM能耗(pJ)": vals["dram_energy_pj"] * energy_scale,
            "SRAM能耗(pJ)": vals.get("sram_energy_pj", 0.0) * energy_scale,
            "总能耗(pJ)": vals["total_energy_pj"] * energy_scale,
        })
        # 如需保留算子平均功耗，按缩放后能量/时长计算；否则可移除此列
        total_ms = op_time_ms.get(op, None)
        if total_ms and total_ms > 0:
            row["平均功耗(W)"] = (vals["total_energy_pj"] * energy_scale) / total_ms * 1e-9
        else:
            row["平均功耗(W)"] = None
        energy_rows.append(row)
    total_row = energy_results.get("__total__", {})
    energy_summary_df = pd.DataFrame(energy_rows)
    # 计算全流程平均功耗：总能量 / (prefill+decode 总时长)
    try:
        # 从 summary_data 中找出该模型的 prefill 与 decode 汇总时长
        total_time_ms = 0.0
        for item in all_data:
            pass
        # 如果 summary_df 已经包含单模型的两行汇总，可直接求和
        if not summary_df.empty and '总延时(ms)' in summary_df.columns:
            total_time_ms = float(summary_df['总延时(ms)'].sum())
        total_energy_pj = float(total_row.get("total_energy_pj", 0.0)) * energy_scale
        avg_power_w = total_energy_pj / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    except Exception:
        avg_power_w = 0.0
    # 仅输出平均功耗，不再展示各部分能耗分项
    # 计算各部分平均功耗（W）
    mac_power_w = total_row.get("mac_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    eltwise_power_w = total_row.get("eltwise_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    noc_power_w = total_row.get("noc_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    pcie_power_w = total_row.get("pcie_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    dram_power_w = total_row.get("dram_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    sram_power_w = total_row.get("sram_energy_pj", 0.0) * energy_scale / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    sram_power_w += getattr(energy_table, 'sram_leakage_power', 0.0)

    total_power_w = mac_power_w + eltwise_power_w + noc_power_w + pcie_power_w + dram_power_w + sram_power_w

    energy_total_df = pd.DataFrame([
        {
            "MAC功耗(W)": mac_power_w,
            "Vector功耗(W)": eltwise_power_w,
            "NoC功耗(W)": noc_power_w,
            "PCIe功耗(W)": pcie_power_w,
            "DRAM访问功耗(W)": dram_power_w,
            "SRAM功耗(W)": sram_power_w,
            "总功耗(W)": total_power_w,
        }
    ])
    
    # 计算时间窗口功耗
    window_size_ms = 50.0  # 时间窗口大小
    
    # 分离prefill和decode的算子能耗数据
    # prefill: 不分步长，直接使用
    prefill_energy_data = [r for r in energy_op_rows if r.get('阶段') == 'prefill' and r.get('算子') != '汇总' and 'decode步长' not in r]
    # decode: 使用所有decode步长的数据（每个步长的数据都是独立的，不是累加的）
    decode_energy_data = [r for r in energy_op_rows if r.get('阶段') == 'decode' and r.get('算子') != '汇总' and 'decode步长' in r]
    
    prefill_windows, decode_windows, power_stats = compute_power_from_inference_data(
        prefill_operator_energy=prefill_energy_data,
        decode_operator_energy=decode_energy_data,
        model_config=model_cfg,
        device_config=device,
        batch_size=batch_size,
        window_size_ms=window_size_ms,
        leakage_power_w=getattr(energy_table, 'sram_leakage_power', 0.0)
    )
    
    # 构建时间窗口功耗DataFrame
    window_power_rows = []
    
    # Prefill窗口
    for i, window in enumerate(prefill_windows):
        window_power_rows.append({
            '阶段': 'prefill',
            '窗口ID': i,
            '开始时间(ms)': window.start_ms,
            '结束时间(ms)': window.end_ms,
            '窗口能耗(J)': window.total_energy_pj * 1e-12,
            '平均功耗(W)': window.avg_power_w,
        })
    
    # Decode窗口
    for i, window in enumerate(decode_windows):
        window_power_rows.append({
            '阶段': 'decode',
            '窗口ID': i,
            '开始时间(ms)': window.start_ms,
            '结束时间(ms)': window.end_ms,
            '窗口能耗(J)': window.total_energy_pj * 1e-12,
            '平均功耗(W)': window.avg_power_w,
        })
    
    window_power_df = pd.DataFrame(window_power_rows)
    
    # 构建功耗统计DataFrame
    power_stats_df = pd.DataFrame([power_stats])
    
    # 保存到Excel
    # 处理输出路径：优先使用原路径，不存在则回落到工作区
    default_path = '/Users/ggk/Documents/performance model/model_inference/latency_data.xlsx'
    output_path = default_path
    try:
        out_dir = os.path.dirname(default_path)
        if not os.path.isdir(out_dir):
            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'latency_energy.xlsx'))
    except Exception:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'latency_energy.xlsx'))
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # 1. 延迟数据工作表 - 只包含A-E列（模型、阶段、算子、总延时、算子占比）
            latency_cols = [c for c in ['模型', '阶段', 'decode步长', '算子', '总延时(ms)', '算子占总延时比例'] if c in df.columns]
            latency_df = df[latency_cols].copy()
            latency_df.to_excel(writer, sheet_name='延迟数据', index=False)
            
            # 2. 汇总数据工作表 - 包含首词延迟、总延时和吞吐率等汇总数据
            summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
            # 3. 能耗总计工作表（仅平均功耗）
            energy_total_df.to_excel(writer, sheet_name='能耗总计', index=False)
            # 4. 算子能耗工作表（按阶段/算子输出能耗与平均功耗）
            if not energy_op_df.empty:
                cols = [c for c in ['模型', '阶段', 'decode步长', '算子', '总延时(ms)', '能耗(pJ)', '平均功耗(W)'] if c in energy_op_df.columns]
                energy_op_df[cols].to_excel(writer, sheet_name='算子能耗', index=False)
            
            # 5. 时间窗口功耗工作表
            if not window_power_df.empty:
                window_power_df.to_excel(writer, sheet_name='时间窗口功耗', index=False)
            
            # 6. 功耗统计工作表
            if not power_stats_df.empty:
                power_stats_df.to_excel(writer, sheet_name='功耗统计', index=False)
            
            # 获取工作表对象并设置格式
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # 设置列宽
                worksheet.set_column('A:A', 15)  # 模型
                
                if sheet_name == '延迟数据':
                    # 动态设置列宽与百分比列格式
                    worksheet.set_column('A:F', 16)
                    percent_format = writer.book.add_format({'num_format': '0.00%'})
                    try:
                        headers = latency_df.columns.tolist()
                        if '算子占总延时比例' in headers:
                            idx = headers.index('算子占总延时比例')
                            col_letter = chr(ord('A') + idx)
                            worksheet.set_column(f'{col_letter}:{col_letter}', 16, percent_format)
                    except Exception:
                        pass
                elif sheet_name == '汇总数据':
                    worksheet.set_column('B:I', 18)  # 所有数据列

                    # 添加标题行格式
                    header_format = writer.book.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BD',
                        'border': 1
                    })

                    # 为汇总数据工作表添加标题行格式
                    for col_num, value in enumerate(summary_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                elif sheet_name == '能耗总计':
                    worksheet.set_column('A:G', 18)
                elif sheet_name == '算子能耗':
                    worksheet.set_column('A:G', 18)
                elif sheet_name == '时间窗口功耗':
                    worksheet.set_column('A:A', 12)  # 阶段
                    worksheet.set_column('B:B', 12)  # 窗口ID
                    worksheet.set_column('C:D', 18)  # 时间
                    worksheet.set_column('E:F', 18)  # 能耗与功耗
                elif sheet_name == '功耗统计':
                    worksheet.set_column('A:L', 20)
        print(f"数据已成功保存到: {output_path}")
    except PermissionError:
        # 如果目标文件被占用，使用带时间戳的新文件名
        ts = time.strftime('%Y%m%d_%H%M%S')
        alt_path = os.path.join(os.path.dirname(output_path), f'latency_energy_{ts}.xlsx')
        with pd.ExcelWriter(alt_path, engine='xlsxwriter') as writer:
            latency_df = df[['模型', '阶段', '算子', '总延时(ms)', '算子占总延时比例']].copy()
            latency_cols = [c for c in ['模型', '阶段', 'decode步长', '算子', '总延时(ms)', '算子占总延时比例'] if c in df.columns]
            latency_df = df[latency_cols].copy()
            latency_df.to_excel(writer, sheet_name='延迟数据', index=False)
            summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
            energy_total_df.to_excel(writer, sheet_name='能耗总计', index=False)
            if not energy_op_df.empty:
                cols = [c for c in ['模型', '阶段', 'decode步长', '算子', '总延时(ms)', '能耗(pJ)', '平均功耗(W)'] if c in energy_op_df.columns]
                energy_op_df[cols].to_excel(writer, sheet_name='算子能耗', index=False)
            if not window_power_df.empty:
                window_power_df.to_excel(writer, sheet_name='时间窗口功耗', index=False)
            if not power_stats_df.empty:
                power_stats_df.to_excel(writer, sheet_name='功耗统计', index=False)
        print(f"目标文件被占用，已改存为: {alt_path}")

if __name__ == "__main__":
    main()