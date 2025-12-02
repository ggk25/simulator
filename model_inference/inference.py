import sys
import os
import json
import pandas as pd
import numpy as np
import math
import time
from types import SimpleNamespace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个模块的类
from software_model.GQA import Prefill as GQAPrefill, Decode as GQADecode
from software_model.MoE import Prefill as MoEPrefill, Decode as MoEDecode
from software_model.FFN import Prefill as FFNPrefill, Decode as FFNDecode
from software_model.MLA import Prefill as MLAPrefill, Decode as MLADecode
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
    """灵活的模型类，根据配置组合不同的模块 (按层调用 mapping_and_simulate)"""
    def __init__(self, config, stage="prefill", context_length=None):
        self.config = config
        self.stage = stage
        self.context_length = context_length

        # attention模块选择
        if config.attention_type == "GQA":
            self.attention = (GQAPrefill if stage == "prefill" else GQADecode)(
                datatype=config.datatype,
                **({
                    'hidden_size': config.hidden_size,
                    'head_dim': config.head_dim,
                    'num_attention_heads': config.num_attention_heads,
                    'num_key_value_heads': config.num_key_value_heads
                } if stage == 'prefill' else {
                    'context_lenth': context_length,
                    'hidden_size': config.hidden_size,
                    'head_dim': config.head_dim,
                    'num_attention_heads': config.num_attention_heads,
                    'num_key_value_heads': config.num_key_value_heads
                })
            )
        elif config.attention_type == "MLA":
            self.attention = (MLAPrefill if stage == "prefill" else MLADecode)(
                datatype=config.datatype,
                **({
                    'hiden_states': config.hidden_size,
                    'q_compress_dim': config.q_compress_dim,
                    'qk_rope_dim': config.qk_rope_dim,
                    'kv_compress_dim': config.kv_compress_dim,
                    'n_heads': config.n_heads,
                    'qkv_dim': config.qkv_dim
                } if stage == 'prefill' else {
                    'context_lenth': context_length,
                    'hiden_states': config.hidden_size,
                    'q_compress_dim': config.q_compress_dim,
                    'qk_rope_dim': config.qk_rope_dim,
                    'kv_compress_dim': config.kv_compress_dim,
                    'n_heads': config.n_heads,
                    'qkv_dim': config.qkv_dim
                })
            )
        # FFN / MoE 模块选择
        if config.ffn_type == "FFN":
            self.ffn = (FFNPrefill if stage == "prefill" else FFNDecode)(
                datatype=config.datatype,
                hidden_size=config.hidden_size,
                intermediate_size=config.ffn_intermediate_size
            )
            self.ffn.hidden_size = config.hidden_size
        else:  # MoE
            self.ffn = (MoEPrefill if stage == "prefill" else MoEDecode)(
                datatype=config.datatype,
                **({
                    'hiden_states': config.hidden_size,
                    'experts_dim': config.experts_dim,
                    'shared_experts_count': config.shared_experts_count,
                    'selected_expert_count': config.selected_expert_count,
                    'experts_count': config.experts_count
                } if stage == 'prefill' else {
                    'context_lenth': context_length,
                    'hiden_states': config.hidden_size,
                    'experts_dim': config.experts_dim,
                    'shared_experts_count': config.shared_experts_count,
                    'selected_expert_count': config.selected_expert_count,
                    'experts_count': config.experts_count
                })
            )

    def __call__(self, input_tensor: Tensor):
        return self.ffn(self.attention(input_tensor))

    def run_one_layer(self, device, layer_id: int, micro_batch_id: int, n_layers_per_chip: int):
        """对单层执行 attention + ffn 的 mapping_and_simulate, 返回合并算子延时与能耗"""
        attn_args = (device, layer_id, micro_batch_id)
        ffn_args = (device, layer_id, micro_batch_id, n_layers_per_chip)
        # Prefill FFN 也接受 n_layers_per_chip; GQA/MLA attention 不需要该参数
        attn_latency_list, attn_total, attn_energy = self.attention.mapping_and_simulate(*attn_args)
        ffn_latency_list, ffn_total, ffn_energy = self.ffn.mapping_and_simulate(*ffn_args)
        combined_latency = attn_latency_list + ffn_latency_list
        combined_energy = attn_energy + ffn_energy
        return combined_latency, attn_total + ffn_total, combined_energy


def collect_model_data(model_config, device, batch_size, prefill_lenth, decode_lenth, use_flexible_model=True):
    micro_batch = math.ceil(batch_size / device.n_chip)
    n_layers_per_chip = math.ceil(model_config.layer_count / device.n_chip)
    model_name = model_config.name
    hidden_size = model_config.hidden_size
    datatype = model_config.datatype
    energy_table_local = load_energy_table()
    sram_leakage_w_total = float(getattr(energy_table_local, 'sram_leakage_power', 0.0))

    # Prefill
    prefill_model = FlexibleModel(model_config, stage="prefill")
    prompt = Tensor([micro_batch, prefill_lenth, hidden_size], data_type=datatype)
    _ = prefill_model(prompt)
    prefill_operator_latency_all = []
    prefill_operator_energy_all = []
    total_prefill_cycles = 0.0
    total_prefill_compute_cycles = 0.0
    total_prefill_comm_cycles = 0.0
    for chip in range(device.n_chip):
        for layer in range(n_layers_per_chip):
            lat_list, layer_total, energy_list = prefill_model.run_one_layer(device, layer_id=layer, micro_batch_id=chip, n_layers_per_chip=n_layers_per_chip)
            prefill_operator_latency_all.extend(lat_list)
            prefill_operator_energy_all.extend(energy_list)
            total_prefill_cycles += layer_total
            total_prefill_compute_cycles += sum(x['计算延时'] for x in lat_list if '计算延时' in x)
            total_prefill_comm_cycles += sum(x['通信延时'] for x in lat_list if '通信延时' in x)

    total_prefill_ms = total_prefill_cycles / (device.frequency * 1e6) * 1e3
    total_prefill_compute_ms = total_prefill_compute_cycles / (device.frequency * 1e6) * 1e3
    total_prefill_comm_ms = total_prefill_comm_cycles / (device.frequency * 1e6) * 1e3
    ttft = total_prefill_ms / 1000.0  # 秒

    # 汇总 per-operator (prefill) — 聚合不同层与不同micro-batch的相同算子
    prefill_data = []
    pref_total_cycles = total_prefill_cycles if total_prefill_cycles > 0 else 1.0
    prefill_op_cycles = {}
    prefill_op_gemm_cycles = {}
    prefill_op_eltwise_cycles = {}
    prefill_op_onchip_cycles = {}
    prefill_op_pcie_cycles = {}
    for op in prefill_operator_latency_all:
        op_name = op['operator']
        prefill_op_cycles[op_name] = prefill_op_cycles.get(op_name, 0.0) + op['总延时']
        prefill_op_gemm_cycles[op_name] = prefill_op_gemm_cycles.get(op_name, 0.0) + op.get('GEMM延时', 0.0)
        prefill_op_eltwise_cycles[op_name] = prefill_op_eltwise_cycles.get(op_name, 0.0) + op.get('ElementWise延时', 0.0)
        prefill_op_onchip_cycles[op_name] = prefill_op_onchip_cycles.get(op_name, 0.0) + op.get('片上通信延时', 0.0)
        prefill_op_pcie_cycles[op_name] = prefill_op_pcie_cycles.get(op_name, 0.0) + op.get('PCIe延时', 0.0)
    for op_name, total_cycles in prefill_op_cycles.items():
        total_ms = total_cycles / (device.frequency * 1e6) * 1e3
        ratio = total_cycles / pref_total_cycles
        gemm_cycles = prefill_op_gemm_cycles.get(op_name, 0.0)
        eltwise_cycles = prefill_op_eltwise_cycles.get(op_name, 0.0)
        onchip_cycles = prefill_op_onchip_cycles.get(op_name, 0.0)
        pcie_cycles = prefill_op_pcie_cycles.get(op_name, 0.0)
        gemm_ms = gemm_cycles / (device.frequency * 1e6) * 1e3
        eltwise_ms = eltwise_cycles / (device.frequency * 1e6) * 1e3
        onchip_ms = onchip_cycles / (device.frequency * 1e6) * 1e3
        pcie_ms = pcie_cycles / (device.frequency * 1e6) * 1e3
        compute_ms = (gemm_cycles + eltwise_cycles) / (device.frequency * 1e6) * 1e3
        comm_ms = (onchip_cycles + pcie_cycles) / (device.frequency * 1e6) * 1e3
        prefill_data.append({
            '模型': model_name,
            '阶段': 'prefill',
            '算子': op_name,
            '计算延时(ms)': compute_ms,
            '通信延时(ms)': comm_ms,
            '总延时(ms)': total_ms,
            '算子占总延时比例': ratio,
            'GEMM延时(ms)': gemm_ms,
            'ElementWise延时(ms)': eltwise_ms,
            '片上通信延时(ms)': onchip_ms,
            'PCIe延时(ms)': pcie_ms
        })

    # Prefill energy rows
    prefill_energy_rows = []
    for e in prefill_operator_energy_all:
        op_name = e['operator']
        total_cycles = float(e.get('总延时', 0.0))
        total_ms = total_cycles / (device.frequency * 1e6) * 1e3
        logic_pj = float(e.get('logic能耗', 0.0))
        dram_pj = float(e.get('DRAM能耗', 0.0))
        logic_power_w = (logic_pj / total_ms * 1e-9 + sram_leakage_w_total) if total_ms > 0 else None
        DRAM_power_w = (dram_pj / total_ms * 1e-9) if total_ms > 0 else None
        prefill_energy_rows.append({'模型': model_name, '阶段': 'prefill', '算子': op_name, '总延时(ms)': total_ms, '逻辑能耗(pJ)': logic_pj, 'DRAM能耗(pJ)': dram_pj, 'logic功耗(w)': logic_power_w, 'DRAM功耗(W)': DRAM_power_w})

    # Prefill 分类汇总
    prefill_total_gemm_ms = sum(prefill_op_gemm_cycles.values()) / (device.frequency * 1e6) * 1e3
    prefill_total_eltwise_ms = sum(prefill_op_eltwise_cycles.values()) / (device.frequency * 1e6) * 1e3
    prefill_total_onchip_ms = sum(prefill_op_onchip_cycles.values()) / (device.frequency * 1e6) * 1e3
    prefill_total_pcie_ms = sum(prefill_op_pcie_cycles.values()) / (device.frequency * 1e6) * 1e3
    prefill_data.append({
        '模型': model_name,
        '阶段': 'prefill',
        '算子': '汇总',
        '计算延时(ms)': total_prefill_compute_ms,
        '通信延时(ms)': total_prefill_comm_ms,
        '总延时(ms)': total_prefill_ms,
        '计算延时占比': total_prefill_compute_cycles / pref_total_cycles,
        '通信延时占比': total_prefill_comm_cycles / pref_total_cycles,
        'GEMM延时(ms)': prefill_total_gemm_ms,
        'ElementWise延时(ms)': prefill_total_eltwise_ms,
        '片上通信延时(ms)': prefill_total_onchip_ms,
        'PCIe延时(ms)': prefill_total_pcie_ms,
        'TTFT(s)': ttft
    })

    # Decode
    decode_data = []
    decode_energy_rows = []
    total_decode_cycles = 0.0
    total_decode_compute_cycles = 0.0
    total_decode_comm_cycles = 0.0
    op_stage_cycles = {}
    op_stage_logic_energy = {}
    op_stage_dram_energy = {}

    decode_total_gemm_cycles = 0.0
    decode_total_eltwise_cycles = 0.0
    decode_total_onchip_cycles = 0.0
    decode_total_pcie_cycles = 0.0
    for step in range(1, decode_lenth + 1):
        context_lenth = prefill_lenth + step
        decode_model = FlexibleModel(model_config, stage="decode", context_length=context_lenth)
        input_step = Tensor([micro_batch, 1, hidden_size], data_type=datatype)
        _ = decode_model(input_step)
        step_lat_list_all = []
        step_energy_list_all = []
        step_compute_cycles = 0.0
        step_comm_cycles = 0.0
        step_total_cycles = 0.0
        for chip in range(device.n_chip):
            for layer in range(n_layers_per_chip):
                lat_list, layer_total, energy_list = decode_model.run_one_layer(device, layer_id=layer, micro_batch_id=chip, n_layers_per_chip=n_layers_per_chip)
                step_lat_list_all.extend(lat_list)
                step_energy_list_all.extend(energy_list)
                step_total_cycles += layer_total
                step_compute_cycles += sum(x['计算延时'] for x in lat_list if '计算延时' in x)
                step_comm_cycles += sum(x['通信延时'] for x in lat_list if '通信延时' in x)

        # per-step operator latency rows — 聚合相同算子在不同层与不同micro-batch的延时
        step_op_cycles = {}
        step_op_gemm_cycles = {}
        step_op_eltwise_cycles = {}
        step_op_onchip_cycles = {}
        step_op_pcie_cycles = {}
        for op in step_lat_list_all:
            op_name = op['operator']
            step_op_cycles[op_name] = step_op_cycles.get(op_name, 0.0) + op['总延时']
            step_op_gemm_cycles[op_name] = step_op_gemm_cycles.get(op_name, 0.0) + op.get('GEMM延时', 0.0)
            step_op_eltwise_cycles[op_name] = step_op_eltwise_cycles.get(op_name, 0.0) + op.get('ElementWise延时', 0.0)
            step_op_onchip_cycles[op_name] = step_op_onchip_cycles.get(op_name, 0.0) + op.get('片上通信延时', 0.0)
            step_op_pcie_cycles[op_name] = step_op_pcie_cycles.get(op_name, 0.0) + op.get('PCIe延时', 0.0)
        for op_name, cycles in step_op_cycles.items():
            ms = cycles / (device.frequency * 1e6) * 1e3
            ratio = cycles / step_total_cycles if step_total_cycles > 0 else 0
            gemm_cycles = step_op_gemm_cycles.get(op_name, 0.0)
            eltwise_cycles = step_op_eltwise_cycles.get(op_name, 0.0)
            onchip_cycles = step_op_onchip_cycles.get(op_name, 0.0)
            pcie_cycles = step_op_pcie_cycles.get(op_name, 0.0)
            gemm_ms = gemm_cycles / (device.frequency * 1e6) * 1e3
            eltwise_ms = eltwise_cycles / (device.frequency * 1e6) * 1e3
            onchip_ms = onchip_cycles / (device.frequency * 1e6) * 1e3
            pcie_ms = pcie_cycles / (device.frequency * 1e6) * 1e3
            compute_ms = (gemm_cycles + eltwise_cycles) / (device.frequency * 1e6) * 1e3
            comm_ms = (onchip_cycles + pcie_cycles) / (device.frequency * 1e6) * 1e3
            decode_data.append({
                '模型': model_name,
                '阶段': 'decode',
                'decode步长': step,
                '算子': op_name,
                '计算延时(ms)': compute_ms,
                '通信延时(ms)': comm_ms,
                '总延时(ms)': ms,
                '算子占总延时比例': ratio,
                'GEMM延时(ms)': gemm_ms,
                'ElementWise延时(ms)': eltwise_ms,
                '片上通信延时(ms)': onchip_ms,
                'PCIe延时(ms)': pcie_ms
            })
            decode_total_gemm_cycles += gemm_cycles
            decode_total_eltwise_cycles += eltwise_cycles
            decode_total_onchip_cycles += onchip_cycles
            decode_total_pcie_cycles += pcie_cycles

        # per-step energy rows
        for e in step_energy_list_all:
            op_name = e['operator']
            cycles = float(e.get('总延时', 0.0))
            ms = cycles / (device.frequency * 1e6) * 1e3
            logic_pj = float(e.get('logic能耗', 0.0))
            dram_pj = float(e.get('DRAM能耗', 0.0))
            logic_power_w = (logic_pj / ms * 1e-9 + sram_leakage_w_total) if ms > 0 else None
            DRAM_power_w = (dram_pj / ms * 1e-9) if ms > 0 else None
            decode_energy_rows.append({'模型': model_name, '阶段': 'decode', 'decode步长': step, '算子': op_name, '总延时(ms)': ms, '逻辑能耗(pJ)': logic_pj, 'DRAM能耗(pJ)': dram_pj, 'logic功耗(W)': logic_power_w, 'DRAM功耗(W)': DRAM_power_w})
            op_stage_logic_energy[op_name] = op_stage_logic_energy.get(op_name, 0.0) + logic_pj
            op_stage_dram_energy[op_name] = op_stage_dram_energy.get(op_name, 0.0) + dram_pj

        for op in step_lat_list_all:
            op_name = op['operator']
            op_stage_cycles[op_name] = op_stage_cycles.get(op_name, 0.0) + op['总延时']

        total_decode_cycles += step_total_cycles
        total_decode_compute_cycles += step_compute_cycles
        total_decode_comm_cycles += step_comm_cycles

    total_decode_ms = total_decode_cycles / (device.frequency * 1e6) * 1e3
    total_decode_compute_ms = total_decode_compute_cycles / (device.frequency * 1e6) * 1e3
    total_decode_comm_ms = total_decode_comm_cycles / (device.frequency * 1e6) * 1e3
    throughput = batch_size * decode_lenth * 1e3 / (total_prefill_ms + total_decode_ms) if (total_prefill_ms + total_decode_ms) > 0 else 0.0
    TBT = total_decode_ms / decode_lenth if decode_lenth > 0 else 0.0

    # stage-level aggregation (decode)
    for op_name, cycles in op_stage_cycles.items():
        ms = cycles / (device.frequency * 1e6) * 1e3
        ratio = cycles / total_decode_cycles if total_decode_cycles > 0 else 0
        logic_pj_total = op_stage_logic_energy.get(op_name, 0.0)
        dram_pj_total = op_stage_dram_energy.get(op_name, 0.0)
        avg_power_w = ((logic_pj_total + dram_pj_total) / ms * 1e-9 + sram_leakage_w_total) if ms > 0 else None
        # 统计阶段分类：从所有 decode_data（非汇总）中搜集对应算子的分类周期
        stage_gemm_cycles = sum(x.get('GEMM延时(ms)', 0.0) * (device.frequency * 1e6) / 1e3 for x in decode_data if x.get('算子') == op_name and 'decode步长' in x)  # revert ms->cycles approximation
        stage_eltwise_cycles = sum(x.get('ElementWise延时(ms)', 0.0) * (device.frequency * 1e6) / 1e3 for x in decode_data if x.get('算子') == op_name and 'decode步长' in x)
        stage_onchip_cycles = sum(x.get('片上通信延时(ms)', 0.0) * (device.frequency * 1e6) / 1e3 for x in decode_data if x.get('算子') == op_name and 'decode步长' in x)
        stage_pcie_cycles = sum(x.get('PCIe延时(ms)', 0.0) * (device.frequency * 1e6) / 1e3 for x in decode_data if x.get('算子') == op_name and 'decode步长' in x)
        gemm_ms = stage_gemm_cycles / (device.frequency * 1e6) * 1e3
        eltwise_ms = stage_eltwise_cycles / (device.frequency * 1e6) * 1e3
        onchip_ms = stage_onchip_cycles / (device.frequency * 1e6) * 1e3
        pcie_ms = stage_pcie_cycles / (device.frequency * 1e6) * 1e3
        # 汇总级别不再重复算子占比, 但保留分类延时
        decode_data.append({
            '模型': model_name,
            '阶段': 'decode',
            '算子': op_name,
            '总延时(ms)': ms,
            '算子占总延时比例': ratio,
            'GEMM延时(ms)': gemm_ms,
            'ElementWise延时(ms)': eltwise_ms,
            '片上通信延时(ms)': onchip_ms,
            'PCIe延时(ms)': pcie_ms
        })
        decode_energy_rows.append({'模型': model_name, '阶段': 'decode', '算子': op_name, '总延时(ms)': ms, '逻辑能耗(pJ)': logic_pj_total, 'DRAM能耗(pJ)': dram_pj_total, '总能耗(pJ)': logic_pj_total + dram_pj_total, '平均功耗(W)': avg_power_w})

    decode_data.append({
        '模型': model_name,
        '阶段': 'decode',
        '算子': '汇总',
        '计算延时(ms)': total_decode_compute_ms,
        '通信延时(ms)': total_decode_comm_ms,
        '总延时(ms)': total_decode_ms,
        '计算延时占比': total_decode_compute_cycles / (total_decode_cycles if total_decode_cycles > 0 else 1.0),
        '通信延时占比': total_decode_comm_cycles / (total_decode_cycles if total_decode_cycles > 0 else 1.0),
        'GEMM延时(ms)': decode_total_gemm_cycles / (device.frequency * 1e6) * 1e3,
        'ElementWise延时(ms)': decode_total_eltwise_cycles / (device.frequency * 1e6) * 1e3,
        '片上通信延时(ms)': decode_total_onchip_cycles / (device.frequency * 1e6) * 1e3,
        'PCIe延时(ms)': decode_total_pcie_cycles / (device.frequency * 1e6) * 1e3,
        '吞吐率(tokens/s)': throughput,
        '生成速度(tokens/s)': 1000 / (TBT if TBT > 0 else 1.0)
    })

    prefill_operators = [x for x in prefill_data if x.get('算子') != '汇总']
    prefill_summary = [x for x in prefill_data if x.get('算子') == '汇总']
    decode_operators = [x for x in decode_data if x.get('算子') != '汇总']
    decode_summary = [x for x in decode_data if x.get('算子') == '汇总']

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

    # 1) Prefill：按算子聚合
    # 2) Decode：按 (decode步长, 算子) 聚合
    # 保留原始 energy_all_rows 用于时间窗口功耗计算
    leakage_power = getattr(load_energy_table(), 'sram_leakage_power', 0.0)
    aggregated_rows = []
    # Prefill 聚合
    prefill_raw = [r for r in energy_all_rows if r.get('阶段') == 'prefill' and r.get('算子') != '汇总' and 'decode步长' not in r]
    prefill_group = {}
    for r in prefill_raw:
        op = r['算子']
        g = prefill_group.setdefault(op, {'模型': model_cfg.name, '阶段': 'prefill', '算子': op, '总延时(ms)': 0.0, '逻辑能耗(pJ)': 0.0, 'DRAM能耗(pJ)': 0.0})
        g['总延时(ms)'] += float(r.get('总延时(ms)', 0.0))
        g['逻辑能耗(pJ)'] += float(r.get('逻辑能耗(pJ)', 0.0))
        g['DRAM能耗(pJ)'] += float(r.get('DRAM能耗(pJ)', 0.0))
    for op, vals in prefill_group.items():
        ms = vals['总延时(ms)'] if vals['总延时(ms)'] > 0 else 1e-9
        logic_pj = vals['逻辑能耗(pJ)']
        dram_pj = vals['DRAM能耗(pJ)']
        total_pj = logic_pj + dram_pj
        logic_power_w = logic_pj / ms * 1e-9 + leakage_power
        dram_power_w = dram_pj / ms * 1e-9
        avg_power_w = total_pj / ms * 1e-9 + leakage_power
        aggregated_rows.append({
            '模型': vals['模型'], '阶段': 'prefill', '算子': op, '总延时(ms)': vals['总延时(ms)'],
            '逻辑能耗(pJ)': logic_pj, 'DRAM能耗(pJ)': dram_pj, '总能耗(pJ)': total_pj,
            '平均功耗(W)': avg_power_w, 'logic功耗(w)': logic_power_w, 'DRAM功耗(W)': dram_power_w
        })
    # Decode 聚合
    decode_raw = [r for r in energy_all_rows if r.get('阶段') == 'decode' and r.get('算子') != '汇总' and 'decode步长' in r]
    decode_group = {}
    for r in decode_raw:
        key = (r.get('decode步长'), r['算子'])
        g = decode_group.setdefault(key, {'模型': model_cfg.name, '阶段': 'decode', 'decode步长': key[0], '算子': key[1], '总延时(ms)': 0.0, '逻辑能耗(pJ)': 0.0, 'DRAM能耗(pJ)': 0.0})
        g['总延时(ms)'] += float(r.get('总延时(ms)', 0.0))
        g['逻辑能耗(pJ)'] += float(r.get('逻辑能耗(pJ)', 0.0))
        g['DRAM能耗(pJ)'] += float(r.get('DRAM能耗(pJ)', 0.0))
    for (step_id, op), vals in decode_group.items():
        ms = vals['总延时(ms)'] if vals['总延时(ms)'] > 0 else 1e-9
        logic_pj = vals['逻辑能耗(pJ)']
        dram_pj = vals['DRAM能耗(pJ)']
        total_pj = logic_pj + dram_pj
        logic_power_w = logic_pj / ms * 1e-9 + leakage_power
        dram_power_w = dram_pj / ms * 1e-9
        avg_power_w = total_pj / ms * 1e-9 + leakage_power
        aggregated_rows.append({
            '模型': vals['模型'], '阶段': 'decode', 'decode步长': step_id, '算子': op, '总延时(ms)': vals['总延时(ms)'],
            '逻辑能耗(pJ)': logic_pj, 'DRAM能耗(pJ)': dram_pj, '总能耗(pJ)': total_pj,
            '平均功耗(W)': avg_power_w, 'logic功耗(w)': logic_power_w, 'DRAM功耗(W)': dram_power_w
        })
    aggregated_energy_op_df = pd.DataFrame(aggregated_rows)
    summary_df = pd.DataFrame(summary_data)
    # 能耗统计
    energy_table = load_energy_table()
    power_counter = get_global_power_counter()
    energy_results = power_counter.compute_energy(energy_table)
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
            "MAC能耗(pJ)": vals["mac_energy_pj"],
            "逐元素能耗(pJ)": vals["eltwise_energy_pj"],
            "NoC能耗(pJ)": vals["noc_energy_pj"],
            "PCIe能耗(pJ)": vals["pcie_energy_pj"],
            "DRAM能耗(pJ)": vals["dram_energy_pj"],
            "SRAM能耗(pJ)": vals.get("sram_energy_pj", 0.0),
            "总能耗(pJ)": vals["total_energy_pj"],
        })
        # 如需保留算子平均功耗，按缩放后能量/时长计算；否则可移除此列
        total_ms = op_time_ms.get(op, None)
        if total_ms and total_ms > 0:
            row["平均功耗(W)"] = vals["total_energy_pj"] / total_ms * 1e-9
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
        total_energy_pj = float(total_row.get("total_energy_pj", 0.0))
        avg_power_w = total_energy_pj / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    except Exception:
        avg_power_w = 0.0
    # 计算各部分平均功耗（W）
    mac_power_w = total_row.get("mac_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    eltwise_power_w = total_row.get("eltwise_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    noc_power_w = total_row.get("noc_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    pcie_power_w = total_row.get("pcie_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    dram_power_w = total_row.get("dram_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
    sram_power_w = total_row.get("sram_energy_pj", 0.0) / total_time_ms * 1e-9 if total_time_ms > 0 else 0.0
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
    
    simple_model_cfg = SimpleNamespace(layer_count=1)
    simple_device_cfg = SimpleNamespace(n_chip=1)
    prefill_windows, decode_windows, power_stats = compute_power_from_inference_data(
        prefill_operator_energy=prefill_energy_data,
        decode_operator_energy=decode_energy_data,
        model_config=simple_model_cfg,
        device_config=simple_device_cfg,
        batch_size=batch_size,
        window_size_ms=window_size_ms,
        leakage_power_w=getattr(energy_table, 'sram_leakage_power', 0.0)
    )
    
    # 构建时间窗口功耗DataFrame
    window_power_rows = []
    
    # Prefill窗口（加入逻辑/DRAM细分）
    for i, window in enumerate(prefill_windows):
        window_power_rows.append({
            '阶段': 'prefill',
            '窗口ID': i,
            '开始时间(ms)': window.start_ms,
            '结束时间(ms)': window.end_ms,
            '窗口能耗(J)': (window.logic_energy_pj + window.dram_energy_pj) * 1e-12,
            '平均功耗(W)': window.avg_power_w,
            '逻辑能耗(pJ)': window.logic_energy_pj,
            'DRAM能耗(pJ)': window.dram_energy_pj,
            '逻辑平均功耗(W)': window.avg_logic_power_w,
            'DRAM平均功耗(W)': window.avg_dram_power_w,
        })
    
    # Decode窗口（加入逻辑/DRAM细分）
    for i, window in enumerate(decode_windows):
        window_power_rows.append({
            '阶段': 'decode',
            '窗口ID': i,
            '开始时间(ms)': window.start_ms,
            '结束时间(ms)': window.end_ms,
            '窗口能耗(J)': (window.logic_energy_pj + window.dram_energy_pj) * 1e-12,
            '平均功耗(W)': window.avg_power_w,
            '逻辑能耗(pJ)': window.logic_energy_pj,
            'DRAM能耗(pJ)': window.dram_energy_pj,
            '逻辑平均功耗(W)': window.avg_logic_power_w,
            'DRAM平均功耗(W)': window.avg_dram_power_w,
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
            latency_cols = [c for c in ['模型', '阶段', 'decode步长', '算子', '计算延时(ms)', '通信延时(ms)', '总延时(ms)', '算子占总延时比例',
                                        'GEMM延时(ms)', 'ElementWise延时(ms)', '片上通信延时(ms)', 'PCIe延时(ms)'] if c in df.columns]
            latency_df = df[latency_cols].copy()
            latency_df.to_excel(writer, sheet_name='延迟数据', index=False)
            
            # 2. 汇总数据工作表 - 包含首词延迟、总延时和吞吐率等汇总数据
            summary_df.to_excel(writer, sheet_name='汇总数据', index=False)
            # 3. 能耗总计工作表（仅平均功耗）
            energy_total_df.to_excel(writer, sheet_name='能耗总计', index=False)
            # 4. 算子能耗工作表（按阶段/算子输出能耗与平均功耗，包含DRAM能耗）
            if not aggregated_energy_op_df.empty:
                cols = [c for c in [
                    '模型', '阶段', 'decode步长', '算子', '总延时(ms)',
                    '逻辑能耗(pJ)', 'DRAM能耗(pJ)', '总能耗(pJ)',
                    '平均功耗(W)', 'logic功耗(w)', 'DRAM功耗(W)'
                ] if c in aggregated_energy_op_df.columns]
                aggregated_energy_op_df[cols].to_excel(writer, sheet_name='算子能耗', index=False)
            
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
                    worksheet.set_column('E:H', 18)  # 总能耗与平均功耗 + 逻辑/DRAM能耗
                    worksheet.set_column('I:J', 18)  # 逻辑/DRAM平均功耗
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
            if not aggregated_energy_op_df.empty:
                cols = [c for c in [
                    '模型', '阶段', 'decode步长', '算子', '总延时(ms)',
                    '逻辑能耗(pJ)', 'DRAM能耗(pJ)', '总能耗(pJ)',
                    '平均功耗(W)', 'logic功耗(w)', 'DRAM功耗(W)'
                ] if c in aggregated_energy_op_df.columns]
                aggregated_energy_op_df[cols].to_excel(writer, sheet_name='算子能耗', index=False)
            if not window_power_df.empty:
                window_power_df.to_excel(writer, sheet_name='时间窗口功耗', index=False)
            if not power_stats_df.empty:
                power_stats_df.to_excel(writer, sheet_name='功耗统计', index=False)
        print(f"目标文件被占用，已改存为: {alt_path}")

if __name__ == "__main__":
    main()