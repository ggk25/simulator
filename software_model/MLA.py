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
from power.dram_power import compute_dram_energy

class Prefill:
    def __init__(self, datatype: DataType ,hiden_states=7168 ,q_compress_dim=1536 ,qk_rope_dim=64 ,kv_compress_dim=576,\
                 n_heads=128 ,qkv_dim=128 ):
        
        self.datatype = datatype
        self.hiden_states = hiden_states
        self.q_compress_dim = q_compress_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_compress_dim = kv_compress_dim
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim

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
        self.qkt_rope = matmul(datatype)       #q * k^T(rope)
        self.q_absorb_mul_ct_kv = matmul(datatype) #q_absorb @ ctkv
        self.rope_add_nope = element_wise_mul_add(datatype) #rope和非rope部分的和加起来
        self.softmax = softmax(datatype)   #softmax(qkT)
        self.s_ctkv = matmul(datatype)     #SV
        self.sv_Wuv = matmul(datatype)     #(SV)*Wuv
        self.sv_concat = reshape(datatype)
        self.linear_o = matmul(datatype)   #linear(o)
        self.att_resadd = element_wise_mul_add(datatype)   #残差相加

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
        return MLA_output
    
    def mapping_and_simulate(self, device:Device ,layer_id: int ,micro_batch_id: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        weight_map = {
            "linear(q_a)": self.WDQ,
            "linear(q_b)": self.WUQ,
            "linear(kv_a)": self.WDKV,
            "q_absorb": self.WUKT,
            "sv_Wuv": self.WUV,
            "linear_o": self.W_o,
        }

        def calc_dram(op_name: str, cycles: float) -> float:
            tensor = weight_map.get(op_name)
            weight_bytes = size(tensor.shape) * self.datatype.word_size if tensor else 0
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, cycles, device.frequency)
            return float(comp.get("total", 0.0))

        def measure(op_name: str, op_obj, comm_fn=lambda sz: 0):
            output_shape = op_obj.output_shape
            out_bytes = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_lat = op_obj.mapping_and_simulate(device)
            comm_lat = comm_fn(out_bytes)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_lat + comm_lat
            operator_latency.append({"operator": op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id,"计算延时": compute_lat, "通信延时": comm_lat, "总延时": total})
            dram_pj = calc_dram(op_name, total)
            operator_energy.append({"operator": op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id, "总延时": total, "logic能耗": energy_delta, "DRAM能耗": dram_pj})
            return total

        # sequence definition: (name, object, communication function)
        steps = [
            ("mla rmsnorm", self.rmsnorm_mla, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("linear(q_a)", self.linear_qa, lambda sz: 0, layer_id, micro_batch_id),
            ("rmsnorm(q)", self.rmsnorm_q, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("linear(q_b)", self.linear_qb, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("rope_q", self.rope_q, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("linear(kv_a)", self.linear_kv_a, lambda sz: 0, layer_id, micro_batch_id),
            ("rmsnorm(kv)", self.rmsnorm_kv, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("rope_k", self.rope_k, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("qkt_rope", self.qkt_rope, lambda sz: 0, layer_id, micro_batch_id),
            ("q_absorb", self.q_absorb, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("q_absorb @ ctkv", self.q_absorb_mul_ct_kv, lambda sz: 0, layer_id, micro_batch_id),
            ("rope_add_nope", self.rope_add_nope, lambda sz: 0, layer_id, micro_batch_id),
            ("softmax", self.softmax, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("s_ctkv", self.s_ctkv, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("sv_Wuv", self.sv_Wuv, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("linear_o", self.linear_o, lambda sz: 0, layer_id, micro_batch_id),
            ("mla_resadd", self.att_resadd, lambda sz: 0, layer_id, micro_batch_id),
        ]

        for name, obj, comm, layer_id, micro_batch_id in steps:
            total_latency += measure(name, obj, comm, layer_id, micro_batch_id)

        return operator_latency, total_latency, operator_energy
    
class Decode:
    def __init__(self, datatype: DataType ,context_lenth ,hiden_states=7168 ,q_compress_dim=1536 ,qk_rope_dim=64 ,kv_compress_dim=576,\
                 n_heads=128 ,qkv_dim=128):
        
        self.datatype = datatype
        self.hiden_states = hiden_states
        self.context_lenth = context_lenth
        self.q_compress_dim = q_compress_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_compress_dim = kv_compress_dim
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim

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
        return MLA_output
    
    def mapping_and_simulate(self, device:Device ,layer_id: int ,micro_batch_id: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        weight_map = {
            "linear(q_a)": self.WDQ,
            "linear(q_b)": self.WUQ,
            "linear(kv_a)": self.WDKV,
            "q_absorb": self.WUKT,
            "sv_Wuv": self.WUV,
            "linear_o": self.W_o,
        }

        def calc_dram(op_name: str, cycles: float) -> float:
            tensor = weight_map.get(op_name)
            weight_bytes = size(tensor.shape) * self.datatype.word_size if tensor else 0
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, cycles, device.frequency)
            return float(comp.get("total", 0.0))

        def measure(op_name: str, op_obj, comm_fn=lambda sz: 0):
            output_shape = op_obj.output_shape
            out_bytes = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_lat = op_obj.mapping_and_simulate(device)
            comm_lat = comm_fn(out_bytes)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_lat + comm_lat
            operator_latency.append({"operator": op_name, "layer_id": layer_id, "micro_batch_id": micro_batch_id, "计算延时": compute_lat, "通信延时": comm_lat, "总延时": total})
            dram_pj = calc_dram(op_name, total)
            operator_energy.append({"operator": op_name, 'micro_batch_id': micro_batch_id, 'layer_id': layer_id, "总延时": total, "logic能耗": energy_delta, "DRAM能耗": dram_pj})
            return total

        steps = [
            ("mla rmsnorm", self.rmsnorm_mla, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("linear(q_a)", self.linear_qa, lambda sz: 0, layer_id, micro_batch_id),
            ("rmsnorm(q)", self.rmsnorm_q, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("linear(q_b)", self.linear_qb, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("rope_q", self.rope_q, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("linear(kv_a)", self.linear_kv_a, lambda sz: 0, layer_id, micro_batch_id),
            ("rmsnorm(kv)", self.rmsnorm_kv, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("rope_k", self.rope_k, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("qkt_rope", self.qkt_rope, lambda sz: 0, layer_id, micro_batch_id),
            ("q_absorb", self.q_absorb, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("q_absorb @ ctkv", self.q_absorb_mul_ct_kv, lambda sz: 0, layer_id, micro_batch_id),
            ("rope_add_nope", self.rope_add_nope, lambda sz: 0, layer_id, micro_batch_id),
            ("softmax", self.softmax, lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id, micro_batch_id),
            ("s_ctkv", self.s_ctkv, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("sv_Wuv", self.sv_Wuv, lambda sz: multicast(device, sz), layer_id, micro_batch_id),
            ("linear_o", self.linear_o, lambda sz: 0, layer_id, micro_batch_id),
            ("mla_resadd", self.att_resadd, lambda sz: 0, layer_id, micro_batch_id),
        ]

        for name, obj, comm, layer_id, micro_batch_id in steps:
            total_latency += measure(name, obj, comm, layer_id, micro_batch_id)

        return operator_latency, total_latency, operator_energy
    