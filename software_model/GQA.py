import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType, Tensor, size
from hardware_model.device import Device
from .gemm import matmul
from .non_gemm import rmsnorm, rope, element_wise_mul_add, softmax
from .misc import reshape, transpose, Concat
from .communication import reduce_multicast, multicast
from power import get_global_power_counter
from power.energy_table import load_energy_table
from power.dram_power import compute_dram_energy
import math

class Prefill:
    def __init__(self, datatype: DataType, hidden_size=5120, head_dim=128, \
                 num_attention_heads=80, num_key_value_heads=8):
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.Wq = Tensor([self.hidden_size, self.num_attention_heads * self.head_dim], self.datatype)
        self.Wk = Tensor([self.hidden_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wv = Tensor([self.hidden_size,self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wo = Tensor([self.num_attention_heads * self.head_dim, self.hidden_size], self.datatype)

        #attention
        self.attn_rmsnorm = rmsnorm(self.datatype)
        self.Q_proj = matmul(self.datatype)
        self.K_proj = matmul(self.datatype)
        self.V_proj = matmul(self.datatype)
        self.Q_reshape = reshape(self.datatype)
        self.K_reshape = reshape(self.datatype)
        self.V_reshape = reshape(self.datatype)
        self.Q_rope = rope(self.datatype)
        self.K_rope = rope(self.datatype)
        self.K_transpose = transpose(self.datatype)
        self.QKT = matmul(self.datatype)
        self.softmax = softmax(self.datatype)
        self.SV = matmul(self.datatype)
        self.SV_reshape = reshape(self.datatype)
        self.O_proj = matmul(self.datatype)
        self.attn_resadd = element_wise_mul_add(self.datatype)

    def __call__(self, input: Tensor) -> Tensor:
        b , s , d = input.shape
        assert d == self.hidden_size
        input = self.attn_rmsnorm(input)
        q = self.Q_proj(input, self.Wq) #(b, s, head_dim * num_attention_heads)
        k = self.K_proj(input, self.Wk) #(b, s, head_dim * num_key_value_heads)
        v = self.V_proj(input, self.Wv) #(b, s, head_dim * num_key_value_heads)
        q = self.Q_reshape(q, [b, self.num_attention_heads, s, self.head_dim]) #(b, num_attention_heads, s, head_dim)
        k = self.K_reshape(k, [b, self.num_key_value_heads, s, self.head_dim]) #(b, num_key_value_heads, s, head_dim)
        v = self.V_reshape(v, [b, self.num_key_value_heads, s, self.head_dim]) #(b, num_key_value_heads, s, head_dim)
        q = self.Q_rope(q) #(b, num_attention_heads, s, head_dim)
        k = self.K_rope(k) #(b, num_key_value_heads, s, head_dim)
        k = self.K_transpose(k ,[0 ,1 ,3, 2]) #(b, num_key_value_heads, head_dim, s)
        qkT = self.QKT(q, k) #(b, num_attention_heads, s, s)
        Score = self.softmax(qkT) #(b, num_attention_heads, s, s)
        SV = self.SV(Score, v) #(b, num_attention_heads, s, head_dim)
        SV = self.SV_reshape(SV, [b, s, self.num_attention_heads * self.head_dim]) #(b, s, head_dim * num_attention_heads)
        O = self.O_proj(SV, self.Wo) #(b, s, hidden_size)
        O = self.attn_resadd(O, input) #(b, s, hidden_size)
        return O
    
    def mapping_and_simulate(self, device: Device, layer_id: int, micro_batch_id: int) :
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, op_obj, duration_cycles: float) -> float:
            # standby energy = standby power (mW) * time (ms) -> pJ (1 mW·ms = 1e6 pJ)
            if getattr(device, 'frequency', 0) <= 0:
                return 0.0
            # active energy: only for weight GEMMs (Q_proj/K_proj/V_proj/O_proj)
            weight_bytes = 0
            if op_name == "Q_proj":
                weight_bytes = size(self.Wq.shape) * self.datatype.word_size
            elif op_name == "K_proj":
                weight_bytes = size(self.Wk.shape) * self.datatype.word_size
            elif op_name == "V_proj":
                weight_bytes = size(self.Wv.shape) * self.datatype.word_size
            elif op_name == "O_proj":
                weight_bytes = size(self.Wo.shape) * self.datatype.word_size
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
            total_pj = float(comp.get('total', 0.0))

            return total_pj

        def measure(op_name, op_obj=None, comm_latency_func=lambda sz: 0, output_shape_override=None, layer_id=None, micro_batch_id=None):
            output_shape = output_shape_override if output_shape_override is not None else op_obj.output_shape
            output_datasize = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_latency = op_obj.mapping_and_simulate(device)
            communication_latency = comm_latency_func(output_datasize)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_latency + communication_latency
            operator_latency.append({'operator': op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id, '计算延时': compute_latency, '通信延时': communication_latency, '总延时': total})
            dram_pj = _calc_dram_energy(op_name, op_obj, total)
            operator_energy.append({'operator': op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total

        total_latency += measure("attn_rmsnorm", self.attn_rmsnorm,
                                lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("Q_proj", self.Q_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("Q_rope", self.Q_rope, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("K_proj", self.K_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("K_rope", self.K_rope, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("V_proj", self.V_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("QKT", self.QKT, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("softmax", self.softmax,
                                lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("SV", self.SV, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("O_proj", self.O_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("attn_resadd", self.attn_resadd, layer_id=layer_id, micro_batch_id=micro_batch_id)

        return operator_latency, total_latency, operator_energy
    
class Decode:
    def __init__(self, datatype: DataType, context_lenth, hidden_size=5120, head_dim=128,
                 num_attention_heads=80, num_key_value_heads=8):
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.context_lenth = context_lenth

        self.Wq = Tensor([self.hidden_size, self.num_attention_heads * self.head_dim], self.datatype)
        self.Wk = Tensor([self.hidden_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wv = Tensor([self.hidden_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wo = Tensor([self.num_attention_heads * self.head_dim, self.hidden_size], self.datatype)

        self.attn_rmsnorm = rmsnorm(self.datatype)
        self.Q_proj = matmul(self.datatype)
        self.K_proj = matmul(self.datatype)
        self.V_proj = matmul(self.datatype)
        self.Q_reshape = reshape(self.datatype)
        self.K_reshape = reshape(self.datatype)
        self.V_reshape = reshape(self.datatype)
        self.K_concat = Concat(self.datatype)
        self.V_concat = Concat(self.datatype)
        self.Q_rope = rope(self.datatype)
        self.K_rope = rope(self.datatype)
        self.K_transpose = transpose(self.datatype)
        self.QKT = matmul(self.datatype)
        self.softmax = softmax(self.datatype)
        self.SV = matmul(self.datatype)
        self.SV_reshape = reshape(self.datatype)
        self.O_proj = matmul(self.datatype)
        self.attn_resadd = element_wise_mul_add(self.datatype)

    def __call__(self, input: Tensor) -> Tensor:
        b , s , d = input.shape
        k_cache = Tensor([b ,self.num_key_value_heads, self.context_lenth ,self.head_dim], self.datatype)
        v_cache = Tensor([b ,self.num_key_value_heads, self.context_lenth ,self.head_dim], self.datatype)
        assert s==1
        assert d == self.hidden_size

        input = self.attn_rmsnorm(input)
        q = self.Q_proj(input, self.Wq) #(b, 1, head_dim * num_attention_heads)
        k = self.K_proj(input, self.Wk) #(b, 1, head_dim * num_key_value_heads)
        v = self.V_proj(input, self.Wv) #(b, 1, head_dim * num_key_value_heads)
        q = self.Q_reshape(q, [b, self.num_attention_heads, s, self.head_dim]) #(b, num_attention_heads, 1, head_dim)
        k = self.K_reshape(k, [b, self.num_key_value_heads, s, self.head_dim]) #(b, num_key_value_heads, 1, head_dim)
        v = self.V_reshape(v, [b, self.num_key_value_heads, s, self.head_dim]) #(b, num_key_value_heads, 1, head_dim)
        v = self.V_concat([v_cache, v], dim=2) #(b, num_key_value_heads, 1+context_lenth, head_dim)
        q = self.Q_rope(q) #(b, num_attention_heads, 1, head_dim)
        k = self.K_rope(k) #(b, num_key_value_heads, 1, head_dim)
        k = self.K_concat([k_cache, k], dim=2) #(b, num_key_value_heads, 1+context_lenth, head_dim)
        k = self.K_transpose(k ,[0 ,1 ,3, 2]) #(b, num_key_value_heads, head_dim, 1+context_lenth)
        qkT = self.QKT(q, k) #(b, num_attention_heads, 1, 1+context_lenth)
        Score = self.softmax(qkT) #(b, num_attention_heads, 1, 1+context_lenth)
        SV = self.SV(Score, v) #(b, num_attention_heads, 1, head_dim)
        SV = self.SV_reshape(SV, [b, s, self.num_attention_heads * self.head_dim]) #(b, 1, head_dim * num_attention_heads)
        O = self.O_proj(SV, self.Wo) #(b, 1, hidden_size)
        O = self.attn_resadd(O, input) #(b, 1, hidden_size)
        return O
    def mapping_and_simulate(self, device: Device, layer_id: int, micro_batch_id: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, op_obj, duration_cycles: float) -> float:
            if getattr(device, 'frequency', 0) <= 0:
                return 0.0
            op_time_ms = duration_cycles / (device.frequency * 1e6) * 1e3
            standby_pj = device.memory.power_ACT_standby * op_time_ms * 1e6
            weight_bytes = 0
            if op_name == "Q_proj":
                weight_bytes = size(self.Wq.shape) * self.datatype.word_size
            elif op_name == "K_proj":
                weight_bytes = size(self.Wk.shape) * self.datatype.word_size
            elif op_name == "V_proj":
                weight_bytes = size(self.Wv.shape) * self.datatype.word_size
            elif op_name == "O_proj":
                weight_bytes = size(self.Wo.shape) * self.datatype.word_size
            if weight_bytes > 0:
                comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
                total_pj = float(comp.get('total', 0.0))
                standby_inside = float(comp.get('standby', 0.0))
                active_pj = max(0.0, total_pj - standby_inside)
            else:
                active_pj = 0.0
            return standby_pj + active_pj

        def measure(op_name, op_obj, comm_latency_func=lambda sz: 0, output_shape_override=None, layer_id=None, micro_batch_id=None):
            output_shape = output_shape_override if output_shape_override is not None else op_obj.output_shape
            output_datasize = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_latency = op_obj.mapping_and_simulate(device)
            communication_latency = comm_latency_func(output_datasize)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_latency + communication_latency
            operator_latency.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '计算延时': compute_latency, '通信延时': communication_latency, '总延时': total})
            dram_pj = _calc_dram_energy(op_name, op_obj, total)
            operator_energy.append({'operator': op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total

        total_latency += measure("attn_rmsnorm", self.attn_rmsnorm,
                                 lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("Q_proj", self.Q_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("Q_rope", self.Q_rope, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("K_proj", self.K_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("K_rope", self.K_rope, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("V_proj", self.V_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("QKT", self.QKT, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("softmax", self.softmax,
                                 lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("SV", self.SV, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("O_proj", self.O_proj, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += measure("attn_resadd", self.attn_resadd, layer_id=layer_id, micro_batch_id=micro_batch_id)

        return operator_latency, total_latency, operator_energy
