import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.device import Device ,device_dict
from .gemm import matmul ,Batched_matmul
from .non_gemm import rmsnorm , rope , element_wise_mul_add , softmax , sigmoid , rank , silu 
from .misc import quantization ,reshape ,split ,transpose ,Concat
from .communication import reduce_multicast ,multicast ,concat ,scatter, p2p
import math
from power import get_global_power_counter
from power.energy_table import load_energy_table

class Prefill():
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
    
    def mapping_and_simulate(self, device:Device):
        #attention
        operator_latency = []
        operator_energy = []
        total_latency = 0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        # attn_rmsnorm
        output_datasize = size(self.attn_rmsnorm.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.attn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "attn_rmsnorm", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # Q_proj
        output_datasize = size(self.Q_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.Q_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "Q_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # Q_rope
        output_datasize = size(self.Q_rope.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.Q_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "Q_rope", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # K_proj
        output_datasize = size(self.K_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.K_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "K_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # K_rope
        output_datasize = size(self.K_rope.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.K_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "K_rope", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # V_proj
        output_datasize = size(self.V_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.V_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "V_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "V_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # QKT
        output_datasize = size(self.QKT.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.QKT.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "QKT", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "QKT", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # softmax
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) +multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "softmax", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # SV
        output_datasize = size(self.SV.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.SV.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency    
        operator_latency_dict = {'operator': "SV", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "SV", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # O_proj
        output_datasize = size(self.O_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.O_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "O_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "O_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # attn_resadd
        output_datasize = size(self.attn_resadd.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.attn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "attn_resadd", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        self.operator_energy = operator_energy
        return operator_latency, total_latency
    
class Decode():
    def __init__(self, datatype: DataType, context_lenth, hidden_size=5120, head_dim=128, \
                 num_attention_heads=80, num_key_value_heads=8):
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.context_lenth = context_lenth

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
        self.K_concat = Concat(self.datatype ,)
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
    def mapping_and_simulate(self, device:Device):
        #attention
        operator_latency = []
        operator_energy = []
        total_latency = 0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        # attn_rmsnorm
        output_datasize = size(self.attn_rmsnorm.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.attn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "attn_rmsnorm", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # Q_proj
        output_datasize = size(self.Q_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.Q_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "Q_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # Q_rope
        output_datasize = size(self.Q_rope.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.Q_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "Q_rope", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # K_proj
        output_datasize = size(self.K_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.K_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "K_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # K_rope
        output_datasize = size(self.K_rope.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.K_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "K_rope", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # V_proj
        output_datasize = size(self.V_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.V_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "V_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "V_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # QKT
        output_datasize = size(self.QKT.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.QKT.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "QKT", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "QKT", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # softmax
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) +multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "softmax", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # SV
        output_datasize = size(self.SV.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.SV.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency    
        operator_latency_dict = {'operator': "SV", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "SV", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # O_proj
        output_datasize = size(self.O_proj.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.O_proj.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "O_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "O_proj", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # attn_resadd
        output_datasize = size(self.attn_resadd.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.attn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "attn_resadd", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        self.operator_energy = operator_energy
        return operator_latency, total_latency
