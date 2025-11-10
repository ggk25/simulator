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

class Prefill():
    def __init__(self, datatype: DataType, hidden_size=5120, head_dim=128, \
                 num_attention_heads=80, num_key_value_heads=8, intermediate_size= 27648):
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size

        self.Wq = Tensor([self.hidden_size, self.num_attention_heads * self.head_dim], self.datatype)
        self.Wk = Tensor([self.hidden_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wv = Tensor([self.hidden_size,self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wo = Tensor([self.num_attention_heads * self.head_dim, self.hidden_size], self.datatype)

        self.W_linear_up = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_size], self.datatype)
        self.W_linear_gate = Tensor([self.hidden_size, self.intermediate_size], self.datatype)

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

        #ffn
        self.ffn_rmsnorm = rmsnorm(self.datatype)
        self.linear_up = matmul(self.datatype)
        self.linear_gate = matmul(self.datatype)
        self.silu = silu(self.datatype)
        self.swiglu_mul = element_wise_mul_add(self.datatype)
        self.linear_down = matmul(self.datatype)
        self.ffn_resadd = element_wise_mul_add(self.datatype)
    
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

        #ffn
        input = self.ffn_rmsnorm(input) #(b, s, hidden_size)
        up = self.linear_up(input, self.W_linear_up)    #(b, s, intermediate_size)
        gate = self.linear_gate(input, self.W_linear_gate)#(b, s, intermediate_size)
        gate = self.silu(gate)
        swiglu = self.swiglu_mul(up, gate)
        down = self.linear_down(swiglu, self.W_linear_down)#(b, s, hidden_size)
        output = self.ffn_resadd(down, input)

        return output
    
    def mapping_and_simulate(self, device:Device):
        #attention
        operator_latency = []
        total_latency = 0

        # attn_rmsnorm
        output_datasize = size(self.attn_rmsnorm.output_shape) * self.datatype.word_size
        compute_latency = self.attn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # Q_proj
        output_datasize = size(self.Q_proj.output_shape) * self.datatype.word_size
        compute_latency = self.Q_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # Q_rope
        output_datasize = size(self.Q_rope.output_shape) * self.datatype.word_size
        compute_latency = self.Q_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # K_proj
        output_datasize = size(self.K_proj.output_shape) * self.datatype.word_size
        compute_latency = self.K_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # K_rope
        output_datasize = size(self.K_rope.output_shape) * self.datatype.word_size
        compute_latency = self.K_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # V_proj
        output_datasize = size(self.V_proj.output_shape) * self.datatype.word_size
        compute_latency = self.V_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "V_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # QKT
        output_datasize = size(self.QKT.output_shape) * self.datatype.word_size
        compute_latency = self.QKT.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "QKT", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # softmax
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) +multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # SV
        output_datasize = size(self.SV.output_shape) * self.datatype.word_size
        compute_latency = self.SV.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency    
        operator_latency_dict = {'operator': "SV", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # O_proj
        output_datasize = size(self.O_proj.output_shape) * self.datatype.word_size
        compute_latency = self.O_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "O_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # attn_resadd
        output_datasize = size(self.attn_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.attn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # ffn_rmsnorm
        output_datasize = size(self.ffn_rmsnorm.output_shape) * self.datatype.word_size
        compute_latency = self.ffn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # linear_up
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        compute_latency = self.linear_up.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # linear_gate
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # silu
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # swiglu_mul
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # linear_down
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # ffn_resadd
        output_datasize = size(self.ffn_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.ffn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 

        return operator_latency, total_latency ,pipeline_latency
    
class Decode():
    def __init__(self, datatype: DataType, context_lenth, hidden_size=5120, head_dim=128, \
                 num_attention_heads=80, num_key_value_heads=8, intermediate_size=27648):
        
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.context_lenth = context_lenth

        self.Wq = Tensor([self.hidden_size, self.num_attention_heads * self.head_dim], self.datatype)
        self.Wk = Tensor([self.hidden_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wv = Tensor([self.hidden_size,self.num_key_value_heads * self.head_dim], self.datatype)
        self.Wo = Tensor([self.num_attention_heads * self.head_dim, self.hidden_size], self.datatype)

        self.W_linear_up = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_size], self.datatype)
        self.W_linear_gate = Tensor([self.hidden_size, self.intermediate_size], self.datatype)

        #attention
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

        #ffn
        self.ffn_rmsnorm = rmsnorm(self.datatype)
        self.linear_up = matmul(self.datatype)
        self.linear_gate = matmul(self.datatype)
        self.silu = silu(self.datatype)
        self.swiglu_mul = element_wise_mul_add(self.datatype)
        self.linear_down = matmul(self.datatype)
        self.ffn_resadd = element_wise_mul_add(self.datatype)

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

        #ffn
        input = self.ffn_rmsnorm(input) #(b, 1, hidden_size)
        up = self.linear_up(input, self.W_linear_up)    #(b, 1, intermediate_size)
        gate = self.linear_gate(input, self.W_linear_gate)#(b, 1, intermediate_size)
        gate = self.silu(gate)
        swiglu = self.swiglu_mul(up, gate)
        down = self.linear_down(swiglu, self.W_linear_down)#(b, 1, hidden_size)
        output = self.ffn_resadd(down, input)

        return output
    
    def mapping_and_simulate(self, device:Device):
        #attention
        operator_latency = []
        total_latency = 0

        # attn_rmsnorm
        output_datasize = size(self.attn_rmsnorm.output_shape) * self.datatype.word_size
        compute_latency = self.attn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # Q_proj
        output_datasize = size(self.Q_proj.output_shape) * self.datatype.word_size
        compute_latency = self.Q_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # Q_rope
        output_datasize = size(self.Q_rope.output_shape) * self.datatype.word_size
        compute_latency = self.Q_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "Q_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # K_proj
        output_datasize = size(self.K_proj.output_shape) * self.datatype.word_size
        compute_latency = self.K_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # K_rope
        output_datasize = size(self.K_rope.output_shape) * self.datatype.word_size
        compute_latency = self.K_rope.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "K_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # V_proj
        output_datasize = size(self.V_proj.output_shape) * self.datatype.word_size
        compute_latency = self.V_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "V_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # QKT
        output_datasize = size(self.QKT.output_shape) * self.datatype.word_size
        compute_latency = self.QKT.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "QKT", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # softmax
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) +multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # SV
        output_datasize = size(self.SV.output_shape) * self.datatype.word_size
        compute_latency = self.SV.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency    
        operator_latency_dict = {'operator': "SV", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # O_proj
        output_datasize = size(self.O_proj.output_shape) * self.datatype.word_size
        compute_latency = self.O_proj.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "O_proj", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # attn_resadd
        output_datasize = size(self.attn_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.attn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "attn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # ffn_rmsnorm
        output_datasize = size(self.ffn_rmsnorm.output_shape) * self.datatype.word_size
        compute_latency = self.ffn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # linear_up
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        compute_latency = self.linear_up.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # linear_gate
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # silu
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # swiglu_mul
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # linear_down
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # ffn_resadd
        output_datasize = size(self.ffn_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.ffn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 

        return operator_latency, total_latency, pipeline_latency


    



