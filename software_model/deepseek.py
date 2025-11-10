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

class Prefill:
    def __init__(self, datatype: DataType ,hiden_states=7168 ,q_compress_dim=1536 ,qk_rope_dim=64 ,kv_compress_dim=576,\
                 n_heads=128 ,qkv_dim=128 ,experts_dim=2048, selected_expert_count=8 ,experts_count= 256):
        
        self.datatype = datatype
        self.hiden_states = hiden_states
        self.q_compress_dim = q_compress_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_compress_dim = kv_compress_dim
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim
        self.experts_dim = experts_dim
        self.selected_expert_count = selected_expert_count
        self.experts_count = experts_count

        #MLA中的权重
        self.hiden_states = hiden_states
        self.WDQ = Tensor([hiden_states ,q_compress_dim],datatype)
        self.WDKV = Tensor([hiden_states ,kv_compress_dim],datatype)
        self.WUQ = Tensor([q_compress_dim ,(qkv_dim + qk_rope_dim)*n_heads],datatype)
        self.WUKT = Tensor([n_heads ,qkv_dim ,kv_compress_dim-qk_rope_dim],datatype)
        self.WUV = Tensor([n_heads, kv_compress_dim - qk_rope_dim ,qkv_dim],datatype)
        self.W_o = Tensor([qkv_dim * n_heads ,hiden_states])

        # MLA
        self.rmsnorm_mla = rmsnorm(datatype)   #RMSNorm
        self.linear_qa = matmul(datatype)  #linear(q_a)
        self.rmsnorm_q = rmsnorm(datatype) #RMSNorm(q)
        self.linear_qb = matmul(datatype)  #linear(q_b)
        self.q_reshape = reshape(datatype)
        self.q_split = split(datatype)
        self.rope_q = rope(datatype)       #rope(q) 
        self.linear_kv_a = matmul(datatype)    #linear(kv_a),kv降维
        self.kv_split = split(datatype)
        self.rmsnorm_kv = rmsnorm(datatype)    #RMSNorm(kv)
        self.rope_k = rope(datatype)           #rope(k)
        self.rope_k_transpose = transpose(datatype)
        self.kv_transpose = transpose(datatype)
        self.q_absorb = matmul(datatype)       #q_absorb
        #没有继续用batched_matmul，因为多头公用了KV，总体计算强度还是很高
        self.qkt_rope = matmul(datatype)       #q * k^T(rope)
        self.q_absorb_mul_ct_kv = matmul(datatype) #q_absorb @ ctkv
        self.rope_add_nope = element_wise_mul_add(datatype) #rope和非rope部分的和加起来
        self.softmax = softmax(datatype)   #softmax(qkT)
        self.s_ctkv = matmul(datatype)     #SV
        self.sv_Wuv = matmul(datatype)     #(SV)*Wuv
        self.sv_concat = reshape(datatype)
        self.linear_o = matmul(datatype)   #linear(o)
        self.att_resadd = element_wise_mul_add(datatype)   #残差相加

        #MoE部分的权重
        self.W_router = Tensor([hiden_states ,experts_count] ,datatype)
        self.W_linear_up = Tensor([hiden_states ,experts_dim] ,datatype)
        self.W_linear_gate = Tensor([hiden_states ,experts_dim] ,datatype)
        self.W_linear_down = Tensor([experts_dim ,hiden_states] ,datatype)

        # MoE
        self.rmsnorm_moe = rmsnorm(datatype)
        self.router = matmul(datatype)  
        self.sigmoid = sigmoid(datatype)
        self.rank = rank(datatype)
        self.rank_add = element_wise_mul_add(datatype)
        self.linear_up = matmul(datatype)
        self.linear_gate = matmul(datatype)
        self.silu = silu(datatype)
        self.swiglu_mul = element_wise_mul_add(datatype)
        self.linear_down = matmul(datatype)
        self.mul = element_wise_mul_add(datatype)
        self.moe_add = element_wise_mul_add(datatype)

    def __call__(self, input:Tensor)-> Tensor:

        b , s , d = input.shape
        assert d == self.hiden_states
        #MLA部分
        input = self.rmsnorm_mla(input) #(b,s,7168)
        Q_compress = self.linear_qa(input ,self.WDQ)    #(b,s,1536)
        Q_compress = self.rmsnorm_q(Q_compress) #(b,s,1536)
        Q_decompress = self.linear_qb(Q_compress , self.WUQ)    #(b,s,128 * (128 + 64))
        Q_decompress = self.q_reshape(Q_decompress ,[b ,self.n_heads ,s ,(self.qkv_dim + self.qk_rope_dim)]) #(b,128,s,192)
        Q_rope ,Q_non_rope = self.q_split(Q_decompress ,[self.qk_rope_dim ,self.qkv_dim] ,dim= 3)   #(b,128,s,64),(b,128,s,128)
        Q_rope = self.rope_q(Q_rope)
        kv_compress = self.linear_kv_a(input ,self.WDKV)    #(b,s,576)
        k_rope, kv_nope = self.kv_split(kv_compress ,[self.qk_rope_dim ,self.kv_compress_dim-self.qk_rope_dim],2)#(b,s,64),(b,s,512)
        k_rope = self.rope_k(k_rope)  #(b, s, 64)
        kv_nope = self.rmsnorm_kv(kv_nope)
        k_rope = self.rope_k_transpose(k_rope ,[0,2,1])
        kv_transpose = self.kv_transpose(kv_nope ,[0,2,1])
        qkT_rope = self.qkt_rope(Q_rope, k_rope)    #(b, 128 ,s ,s)
        q_absorb = self.q_absorb(Q_non_rope, self.WUKT) #(b,128,s,512)
        qkT = self.q_absorb_mul_ct_kv(q_absorb ,kv_transpose)#(b,128,s,s)
        qkT = self.rope_add_nope(qkT,qkT_rope)#(b,128,s,s)
        score = self.softmax(qkT)
        sv = self.s_ctkv(score ,kv_nope)
        v_absorb = self.sv_Wuv(sv ,self.WUV)    #(b,128,s,128)
        v_absorb = self.sv_concat(v_absorb ,[b,s,self.n_heads * self.qkv_dim])   #(b,s,16384)
        v_absorb_mul_Wo = self.linear_o(v_absorb ,self.W_o) #(b,s,7168)
        MLA_output = self.att_resadd(v_absorb_mul_Wo ,input)

        #MoE
        MLA_output = self.rmsnorm_moe(MLA_output)
        router = self.router(MLA_output,self.W_router)    #(b,s,256)
        router = self.sigmoid(router)
        router = self.rank(router)  #排序
        router = self.rank_add(router)
        linear_up = self.linear_up(MLA_output ,self.W_linear_up)    #(b,s,2048)
        linear_gate = self.linear_gate(MLA_output ,self.W_linear_gate)
        linear_gate = self.silu(linear_gate)
        linear_up = self.swiglu_mul(linear_up ,linear_gate)
        linear_down = self.linear_down(linear_up ,self.W_linear_down)#(b,s,7168)
        linear_down = self.mul(linear_down)
        linear_down = self.moe_add(linear_down ,linear_down)

        return linear_down
    
    def mapping_and_simulate(self, device:Device ):

        operator_latency = []
        total_latency = 0

        # print("mla rmsnorm")
        output_datasize = size(self.rmsnorm_mla.output_shape) * self.datatype.word_size   #通信时传输的都是fp32
        compute_latency = self.rmsnorm_mla.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency +=compute_latency + communication_latency
        operator_latency_dict = {'operator': "mla rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(q_a)")
        output_datasize = size(self.linear_qa.output_shape) * self.datatype.word_size
        compute_latency = self.linear_qa.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(q_a)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rmsnorm(q)")
        output_datasize = size(self.rmsnorm_q.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_q.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm(q)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(q_b)")
        output_datasize = size(self.linear_qb.output_shape) * self.datatype.word_size
        compute_latency = self.linear_qb.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(q_b)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope_q")
        output_datasize = size(self.rope_q.output_shape) * self.datatype.word_size
        compute_latency = self.rope_q.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_q", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(kv_a)")
        output_datasize = size(self.linear_kv_a.output_shape) * self.datatype.word_size
        compute_latency = self.linear_kv_a.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(kv_a)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rmsnorm(kv)")
        output_datasize = size(self.rmsnorm_kv.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_kv.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm(kv)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope(k)")
        output_datasize = size(self.rope_k.output_shape) * self.datatype.word_size
        compute_latency = self.rope_k.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_k", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("qk^T(rope)")
        output_datasize = size(self.qkt_rope.output_shape) * self.datatype.word_size
        compute_latency = self.qkt_rope.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "qkt_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("q_absorb")
        output_datasize = size(self.q_absorb.output_shape) * self.datatype.word_size
        compute_latency = self.q_absorb.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "q_absorb", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("q_absorb @ ctkv")
        output_datasize = size(self.q_absorb_mul_ct_kv.output_shape) * self.datatype.word_size
        compute_latency = self.q_absorb_mul_ct_kv.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "q_absorb @ ctkv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope_add_nope")
        output_datasize = size(self.rope_add_nope.output_shape) * self.datatype.word_size
        compute_latency = self.rope_add_nope.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_add_nope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("softmax")    #
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sv")
        output_datasize = size(self.s_ctkv.output_shape) * self.datatype.word_size
        compute_latency = self.s_ctkv.mapping_and_simulate(device)
        communication_latency = multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "s_ctkv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sv_Wuv")
        output_datasize = size(self.sv_Wuv.output_shape) * self.datatype.word_size
        compute_latency = self.sv_Wuv.mapping_and_simulate(device)
        communication_latency = multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "sv_Wuv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(o)")
        output_datasize = size(self.linear_o.output_shape) * self.datatype.word_size
        compute_latency = self.linear_o.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_o", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("mla_resadd")
        output_datasize = size(self.att_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.att_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mla_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("moe_rmsnorm")
        output_datasize = size(self.rmsnorm_moe.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_moe.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm_moe", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_router")
        output_datasize = size(self.router.output_shape) * self.datatype.word_size
        compute_latency = self.router.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "router", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sigmoid")
        output_datasize = size(self.sigmoid.output_shape) * self.datatype.word_size
        compute_latency = self.sigmoid.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "sigmoid", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rank_add")
        output_datasize = size(self.rank_add.output_shape) * self.datatype.word_size
        compute_latency = self.rank_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rank_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_up")
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_up.mapping_and_simulate(device)   #需要选中8个专家
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_gate")
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("silu")
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("swiglu_mul")
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = 9 * multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_down")
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # print("mul")
        output_datasize = size(self.mul.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.mul.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("expert_add")
        output_datasize = size(self.moe_add.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.moe_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "moe_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 

        return operator_latency ,total_latency ,pipeline_latency

class Decode:
    def __init__(self, datatype: DataType ,context_lenth ,hiden_states=7168 ,q_compress_dim=1536 ,qk_rope_dim=64 ,kv_compress_dim=576,\
                 n_heads=128 ,qkv_dim=128 ,experts_dim=2048, selected_expert_count=8 ,experts_count= 256):
        
        self.datatype = datatype
        self.hiden_states = hiden_states
        self.q_compress_dim = q_compress_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_compress_dim = kv_compress_dim
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim
        self.experts_dim = experts_dim
        self.selected_expert_count = selected_expert_count
        self.experts_count = experts_count
        self.context_lenth = context_lenth

        #MLA中的权重
        self.hiden_states = hiden_states
        self.WDQ = Tensor([hiden_states ,q_compress_dim],datatype)
        self.WDKV = Tensor([hiden_states ,kv_compress_dim],datatype)
        self.WUQ = Tensor([q_compress_dim ,(qkv_dim + qk_rope_dim)*n_heads],datatype)
        self.WUKT = Tensor([n_heads ,qkv_dim ,kv_compress_dim-qk_rope_dim],datatype)
        self.WUV = Tensor([n_heads, kv_compress_dim - qk_rope_dim ,qkv_dim],datatype)
        self.W_o = Tensor([qkv_dim * n_heads ,hiden_states],datatype)

        # MLA
        self.rmsnorm_mla = rmsnorm(datatype)   #RMSNorm
        self.linear_qa = matmul(datatype)  #linear(q_a)
        self.rmsnorm_q = rmsnorm(datatype) #RMSNorm(q)
        self.linear_qb = matmul(datatype)  #linear(q_b)
        self.q_reshape = reshape(datatype)
        self.q_split = split(datatype)
        self.rope_q = rope(datatype)       #rope(q)
        self.linear_kv_a = matmul(datatype)    #linear(kv_a),kv降维
        self.concat_context = Concat(datatype)
        self.kv_split = split(datatype)
        self.rmsnorm_kv = rmsnorm(datatype)    #RMSNorm(kv)
        self.rope_k = rope(datatype)           #rope(k)
        self.rope_k_transpose = transpose(datatype)
        self.kv_transpose = transpose(datatype)
        self.q_absorb = matmul(datatype)       #q_absorb
        self.qkt_rope = matmul(datatype)       #q * k^T(rope)
        self.q_absorb_mul_ct_kv = matmul(datatype) #q_absorb @ ctkv
        self.rope_add_nope = element_wise_mul_add(datatype) #rope和非rope部分的和加起来
        self.softmax = softmax(datatype)   #softmax(qkT)
        self.s_ctkv = matmul(datatype)     #SV
        self.sv_Wuv = matmul(datatype)     #(SV)*Wuv
        self.sv_concat = reshape(datatype)
        self.linear_o = matmul(datatype)   
        self.att_resadd = element_wise_mul_add(datatype)   #残差相加

        #MoE部分的权重
        self.W_router = Tensor([hiden_states ,experts_count] ,datatype)
        self.W_linear_up = Tensor([hiden_states ,experts_dim] ,datatype)
        self.W_linear_gate = Tensor([hiden_states ,experts_dim] ,datatype)
        self.W_linear_down = Tensor([experts_dim ,hiden_states] ,datatype)

        # MoE
        self.rmsnorm_moe = rmsnorm(datatype)
        self.router = matmul(datatype)
        self.sigmoid = sigmoid(datatype)
        self.rank = rank(datatype)
        self.rank_add = element_wise_mul_add(datatype)
        self.linear_up = matmul(datatype)
        self.linear_gate = matmul(datatype)
        self.silu = silu(datatype)
        self.swiglu_mul = element_wise_mul_add(datatype)
        self.linear_down = matmul(datatype)
        self.mul = element_wise_mul_add(datatype)
        self.moe_add = element_wise_mul_add(datatype)

    def __call__(self, input:Tensor)-> Tensor:

        b , s , d = input.shape
        kv_cache = Tensor([b ,self.context_lenth ,self.kv_compress_dim])
        assert s==1
        assert d == self.hiden_states
        #MLA部分
        input = self.rmsnorm_mla(input) #(b,1,7168)
        Q_compress = self.linear_qa(input ,self.WDQ)    #(b,1,1536)
        Q_compress = self.rmsnorm_q(Q_compress) #(b,1,1536)
        Q_decompress = self.linear_qb(Q_compress , self.WUQ)    #(b,1,128 * (128 + 64))
        Q_decompress = self.q_reshape(Q_decompress ,[b ,self.n_heads ,s ,(self.qkv_dim + self.qk_rope_dim)]) #(b,128,1,192)
        Q_rope ,Q_non_rope = self.q_split(Q_decompress ,[self.qk_rope_dim ,self.qkv_dim] ,dim= 3)   #(b,128,1,64),(b,128,1,128)
        Q_rope = self.rope_q(Q_rope)
        kv_compress = self.linear_kv_a(input ,self.WDKV)    #(b,1,576)
        k_rope, kv_nope = self.kv_split(kv_compress ,[self.qk_rope_dim ,self.kv_compress_dim-self.qk_rope_dim],2)#(b,1,64),(b,1,512)
        k_rope = self.rope_k(k_rope)  #(b, 1, 64)
        kv_nope = self.rmsnorm_kv(kv_nope)
        kv_compress = self.concat_context((kv_compress ,kv_cache) ,1)
        k_rope, kv_nope = self.kv_split(kv_compress ,[self.qk_rope_dim ,self.kv_compress_dim-self.qk_rope_dim],2)#(b,context_lenth+1,64),(b,context_lenth+1,512)
        k_rope = self.rope_k_transpose(k_rope ,[0,2,1])
        kv_transpose = self.kv_transpose(kv_nope ,[0,2,1])
        qkT_rope = self.qkt_rope(Q_rope, k_rope)    #(b, 128 ,1 ,context_lenth + 1)
        q_absorb = self.q_absorb(Q_non_rope, self.WUKT) #(b,128,1,512)
        qkT = self.q_absorb_mul_ct_kv(q_absorb ,kv_transpose)#(b,128,1,context_lenth + 1)
        qkT = self.rope_add_nope(qkT,qkT_rope)#(b,128,1,context_lenth + 1)
        score = self.softmax(qkT)
        sv = self.s_ctkv(score ,kv_nope)
        v_absorb = self.sv_Wuv(sv ,self.WUV)    #(b,128,1,128)
        v_absorb = self.sv_concat(v_absorb ,[b,s,self.n_heads * self.qkv_dim])   #(b,1,16384)
        v_absorb_mul_Wo = self.linear_o(v_absorb ,self.W_o) #(b,1,7168)
        MLA_output = self.att_resadd(v_absorb_mul_Wo ,input)

        #MoE
        MLA_output = self.rmsnorm_moe(MLA_output)
        router = self.router(MLA_output,self.W_router)    #(b,1,256)
        router = self.sigmoid(router)
        router = self.rank(router)  #排序
        router = self.rank_add(router)
        linear_up = self.linear_up(MLA_output ,self.W_linear_up)    #(b,1,2048)
        linear_gate = self.linear_gate(MLA_output ,self.W_linear_gate)
        linear_gate = self.silu(linear_gate)
        linear_up = self.swiglu_mul(linear_up ,linear_gate)
        linear_down = self.linear_down(linear_up ,self.W_linear_down)#(b,1,7168)
        linear_down = self.mul(linear_down)
        linear_down = self.moe_add(linear_down ,linear_down)

        return linear_down
    
    def mapping_and_simulate(self, device:Device ):

        operator_latency = []
        total_latency = 0

        # print("mla rmsnorm")
        output_datasize = size(self.rmsnorm_mla.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_mla.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency +=compute_latency + communication_latency
        operator_latency_dict = {'operator': "mla rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(q_a)")
        output_datasize = size(self.linear_qa.output_shape) * self.datatype.word_size
        compute_latency = self.linear_qa.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(q_a)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rmsnorm(q)")
        output_datasize = size(self.rmsnorm_q.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_q.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm(q)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(q_b)")
        output_datasize = size(self.linear_qb.output_shape) * self.datatype.word_size
        compute_latency = self.linear_qb.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(q_b)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope_q")
        output_datasize = size(self.rope_q.output_shape) * self.datatype.word_size
        compute_latency = self.rope_q.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_q", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(kv_a)")
        output_datasize = size(self.linear_kv_a.output_shape) * self.datatype.word_size
        compute_latency = self.linear_kv_a.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear(kv_a)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rmsnorm(kv)")
        output_datasize = size(self.rmsnorm_kv.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_kv.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm(kv)", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope(k)")
        output_datasize = size(self.rope_k.output_shape) * self.datatype.word_size
        compute_latency = self.rope_k.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_k", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("qk^T(rope)")
        output_datasize = size(self.qkt_rope.output_shape) * self.datatype.word_size
        compute_latency = self.qkt_rope.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "qkt_rope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("q_absorb")
        output_datasize = size(self.q_absorb.output_shape) * self.datatype.word_size
        compute_latency = self.q_absorb.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "q_absorb", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("q_absorb @ ctkv")
        output_datasize = size(self.q_absorb_mul_ct_kv.output_shape) * self.datatype.word_size
        compute_latency = self.q_absorb_mul_ct_kv.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "q_absorb @ ctkv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rope_add_nope")
        output_datasize = size(self.rope_add_nope.output_shape) * self.datatype.word_size
        compute_latency = self.rope_add_nope.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rope_add_nope", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("softmax")  
        output_datasize = size(self.softmax.output_shape) * self.datatype.word_size
        compute_latency = self.softmax.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "softmax", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sv")
        output_datasize = size(self.s_ctkv.output_shape) * self.datatype.word_size
        compute_latency = self.s_ctkv.mapping_and_simulate(device)
        communication_latency = multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "s_ctkv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sv_Wuv")
        output_datasize = size(self.sv_Wuv.output_shape) * self.datatype.word_size
        compute_latency = self.sv_Wuv.mapping_and_simulate(device)
        communication_latency = multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "sv_Wuv", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear(o)")
        output_datasize = size(self.linear_o.output_shape) * self.datatype.word_size
        compute_latency = self.linear_o.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_o", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("mla_resadd")
        output_datasize = size(self.att_resadd.output_shape) * self.datatype.word_size
        compute_latency = self.att_resadd.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mla_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("moe_rmsnorm")
        output_datasize = size(self.rmsnorm_moe.output_shape) * self.datatype.word_size
        compute_latency = self.rmsnorm_moe.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rmsnorm_moe", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_router")
        output_datasize = size(self.router.output_shape) * self.datatype.word_size
        compute_latency = self.router.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "router", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("sigmoid")
        output_datasize = size(self.sigmoid.output_shape) * self.datatype.word_size
        compute_latency = self.sigmoid.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "sigmoid", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("rank_add")
        output_datasize = size(self.rank_add.output_shape) * self.datatype.word_size
        compute_latency = self.rank_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "rank_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_up")
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_up.mapping_and_simulate(device)   #需要选中8个专家
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_gate")
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("silu")
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("swiglu_mul")
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = 9 * multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_down")
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # print("mul")
        output_datasize = size(self.mul.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.mul.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("expert_add")
        output_datasize = size(self.moe_add.output_shape) * self.datatype.word_size
        compute_latency = 9 * self.moe_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "moe_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        pipeline_latency = concat(device,output_datasize) + p2p(device,output_datasize) + scatter(device,output_datasize)


        return operator_latency ,total_latency ,pipeline_latency

'''
device = device_dict["D37x"]
input = Tensor([8,1,7168] ,data_type_dict["fp8"])
decode = Decode(data_type_dict["fp8"] ,1024)
output = decode(input)
latency = decode.mapping_and_simulate(device)
'''