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
    def __init__(self, datatype: DataType ,hiden_states=7168 ,experts_dim=2048, shared_experts_count=1, selected_expert_count=8 ,experts_count= 256):

        self.datatype = datatype
        self.hiden_states = hiden_states
        self.experts_dim = experts_dim
        self.shared_experts_count = shared_experts_count
        self.selected_expert_count = selected_expert_count
        self.experts_count = experts_count

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
        #MoE
        MLA_output = self.rmsnorm_moe(input)
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
    
    def mapping_and_simulate(self, device:Device, layer_id: int, micro_batch_id: int, n_layers_per_chip: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        weight_map = {
            "router": self.W_router,
            "linear_up": self.W_linear_up,
            "linear_gate": self.W_linear_gate,
            "linear_down": self.W_linear_down,
        }

        def calc_dram(op_name: str, cycles: float, experts_mult: int = 1) -> float:
            tensor = weight_map.get(op_name)
            weight_bytes = size(tensor.shape) * self.datatype.word_size if tensor else 0
            if experts_mult > 1 and tensor:
                weight_bytes *= experts_mult
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, cycles, device.frequency)
            return float(comp.get('total', 0.0))

        def measure(op_name: str, op_obj, comm_fn=lambda sz: 0, experts_mult: int = 1, layer_id: int = None, micro_batch_id: int = None):
            out_shape = op_obj.output_shape
            out_bytes = size(out_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_lat = 0.0
            reps = experts_mult if experts_mult > 1 else 1
            for _ in range(reps):
                compute_lat += op_obj.mapping_and_simulate(device)
            comm_lat = comm_fn(out_bytes, reps)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_lat + comm_lat
            operator_latency.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '计算延时': compute_lat, '通信延时': comm_lat, '总延时': total})
            dram_pj = calc_dram(op_name, total, experts_mult)
            operator_energy.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total, out_bytes

        experts = (self.shared_experts_count + self.selected_expert_count)

        def comm_zero(sz, reps=1):
            return 0

        def comm_reduce_then_mc(sz, reps=1):
            return reduce_multicast(device) + multicast(device, sz)

        def comm_mc(sz, reps=1):
            return multicast(device, sz) * reps

        # rmsnorm_moe
        t, last_bytes = measure('rmsnorm_moe', self.rmsnorm_moe, comm_reduce_then_mc, layer_id, micro_batch_id)
        total_latency += t
        # router
        t, last_bytes = measure('router', self.router, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # sigmoid
        t, last_bytes = measure('sigmoid', self.sigmoid, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # rank_add
        t, last_bytes = measure('rank_add', self.rank_add, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_up
        t, last_bytes = measure('linear_up', self.linear_up, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_gate
        t, last_bytes = measure('linear_gate', self.linear_gate, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # silu
        t, last_bytes = measure('silu', self.silu, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # swiglu_mul
        t, last_bytes = measure('swiglu_mul', self.swiglu_mul, comm_mc, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_down
        t, last_bytes = measure('linear_down', self.linear_down, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # mul
        t, last_bytes = measure('mul', self.mul, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # moe_add
        t, last_bytes = measure('moe_add', self.moe_add, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t

        if( (layer_id +1) % n_layers_per_chip ==0 and device.n_chip >1):
            pipeline_latency = concat(device, last_bytes) + p2p(device, last_bytes) + scatter(device, last_bytes)
            operator_latency.append({'operator': 'pipeline_communication', 'micro_batch_id': micro_batch_id, '总延时': pipeline_latency})
            total_latency += pipeline_latency

        return operator_latency, total_latency, operator_energy
    
class Decode:
    def __init__(self, datatype: DataType ,context_lenth ,hiden_states=7168 ,experts_dim=2048, shared_experts_count=1, selected_expert_count=8 ,experts_count= 256):
        
        self.datatype = datatype
        self.hiden_states = hiden_states
        self.experts_dim = experts_dim
        self.shared_experts_count = shared_experts_count
        self.selected_expert_count = selected_expert_count
        self.experts_count = experts_count
        self.context_lenth = context_lenth

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
        #MoE
        MLA_output = self.rmsnorm_moe(input)
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
    
    def mapping_and_simulate(self, device:Device, layer_id: int, micro_batch_id: int, n_layers_per_chip: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        weight_map = {
            "router": self.W_router,
            "linear_up": self.W_linear_up,
            "linear_gate": self.W_linear_gate,
            "linear_down": self.W_linear_down,
        }

        def calc_dram(op_name: str, cycles: float, experts_mult: int = 1) -> float:
            tensor = weight_map.get(op_name)
            weight_bytes = size(tensor.shape) * self.datatype.word_size if tensor else 0
            if experts_mult > 1 and tensor:
                weight_bytes *= experts_mult
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, cycles, device.frequency)
            return float(comp.get('total', 0.0))

        def measure(op_name: str, op_obj, comm_fn=lambda sz: 0, experts_mult: int = 1, layer_id: int = None, micro_batch_id: int = None):
            out_shape = op_obj.output_shape
            out_bytes = size(out_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_lat = 0.0
            reps = experts_mult if experts_mult > 1 else 1
            for _ in range(reps):
                compute_lat += op_obj.mapping_and_simulate(device)
            comm_lat = comm_fn(out_bytes, reps)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_lat + comm_lat
            operator_latency.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '计算延时': compute_lat, '通信延时': comm_lat, '总延时': total})
            dram_pj = calc_dram(op_name, total, experts_mult)
            operator_energy.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total, out_bytes

        experts = (self.shared_experts_count + self.selected_expert_count)

        def comm_zero(sz, reps=1):
            return 0

        def comm_reduce_then_mc(sz, reps=1):
            return reduce_multicast(device) + multicast(device, sz)

        def comm_mc(sz, reps=1):
            return multicast(device, sz) * reps

        # rmsnorm_moe
        t, last_bytes = measure('rmsnorm_moe', self.rmsnorm_moe, comm_reduce_then_mc, layer_id, micro_batch_id)
        total_latency += t
        # router
        t, last_bytes = measure('router', self.router, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # sigmoid
        t, last_bytes = measure('sigmoid', self.sigmoid, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # rank_add
        t, last_bytes = measure('rank_add', self.rank_add, comm_zero, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_up
        t, last_bytes = measure('linear_up', self.linear_up, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_gate
        t, last_bytes = measure('linear_gate', self.linear_gate, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # silu
        t, last_bytes = measure('silu', self.silu, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # swiglu_mul
        t, last_bytes = measure('swiglu_mul', self.swiglu_mul, comm_mc, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_down
        t, last_bytes = measure('linear_down', self.linear_down, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # mul
        t, last_bytes = measure('mul', self.mul, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # moe_add
        t, last_bytes = measure('moe_add', self.moe_add, comm_zero, experts, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t

        if( (layer_id +1) % n_layers_per_chip ==0 and device.n_chip >1):
            pipeline_latency = concat(device, last_bytes) + p2p(device, last_bytes) + scatter(device, last_bytes)
            operator_latency.append({'operator': 'pipeline_communication', 'micro_batch_id': micro_batch_id, '总延时': pipeline_latency})
            total_latency += pipeline_latency
        
        return operator_latency, total_latency, operator_energy