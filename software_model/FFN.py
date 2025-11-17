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

class Prefill():
    def __init__(self, datatype: DataType, hidden_size=5120 ,intermediate_size= 27648):
        self.datatype = datatype
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.W_linear_up = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_size], self.datatype)
        self.W_linear_gate = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
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
        #ffn
        input = self.ffn_rmsnorm(input) #(b, s, hidden_size)
        up = self.linear_up(input, self.W_linear_up)    #(b, s, intermediate_size)
        gate = self.linear_gate(input, self.W_linear_gate)#(b, s, intermediate_size)
        gate = self.silu(gate)
        swiglu = self.swiglu_mul(up, gate)
        down = self.linear_down(swiglu, self.W_linear_down)#(b, s, hidden_size)
        output = self.ffn_resadd(down, input)

        return output
    
    def mapping_and_simulate(self, device:Device, layer_id: int, micro_batch_id: int, n_layers_per_chip: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, duration_cycles: float) -> float:
            # Use compute_dram_energy total: includes standby + active. Active only when weights exist.
            weight_bytes = 0
            if op_name == "linear_up":
                weight_bytes = size(self.W_linear_up.shape) * self.datatype.word_size
            elif op_name == "linear_gate":
                weight_bytes = size(self.W_linear_gate.shape) * self.datatype.word_size
            elif op_name == "linear_down":
                weight_bytes = size(self.W_linear_down.shape) * self.datatype.word_size
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
            return float(comp.get('total', 0.0))

        def measure(op_name, op_obj, comm_latency_func=lambda sz: 0, output_shape_override=None, layer_id: int = None, micro_batch_id: int = None):
            output_shape = output_shape_override if output_shape_override is not None else op_obj.output_shape
            output_datasize = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_latency = op_obj.mapping_and_simulate(device)
            communication_latency = comm_latency_func(output_datasize)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_latency + communication_latency
            operator_latency.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '计算延时': compute_latency, '通信延时': communication_latency, '总延时': total})
            dram_pj = _calc_dram_energy(op_name, total)
            operator_energy.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total, output_datasize

        # ffn_rmsnorm
        t, last_out = measure("ffn_rmsnorm", self.ffn_rmsnorm,
                              lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_up
        t, last_out = measure("linear_up", self.linear_up, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_gate
        t, last_out = measure("linear_gate", self.linear_gate, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # silu
        t, last_out = measure("silu", self.silu, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # swiglu_mul
        t, last_out = measure("swiglu_mul", self.swiglu_mul, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)   
        total_latency += t
        # linear_down
        t, last_out = measure("linear_down", self.linear_down, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # ffn_resadd
        t, last_out= measure("ffn_resadd", self.ffn_resadd, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        
        if( (layer_id +1) % n_layers_per_chip ==0):
            pipeline_latency = concat(device ,last_out) + p2p(device ,last_out) + scatter(device ,last_out)
            operator_latency.append({'operator': 'pipeline_communication', 'micro_batch_id': micro_batch_id, '总延时': pipeline_latency})
            total_latency += pipeline_latency
        
        return operator_latency, total_latency, operator_energy
    
class Decode():
    def __init__(self, datatype: DataType, hidden_size=5120, intermediate_size=27648):
        self.datatype = datatype
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.W_linear_up = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_size], self.datatype)
        self.W_linear_gate = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        #ffn
        self.ffn_rmsnorm = rmsnorm(self.datatype)
        self.linear_up = matmul(self.datatype)
        self.linear_gate = matmul(self.datatype)
        self.silu = silu(self.datatype)
        self.swiglu_mul = element_wise_mul_add(self.datatype)
        self.linear_down = matmul(self.datatype)
        self.ffn_resadd = element_wise_mul_add(self.datatype)

    def __call__(self, input: Tensor) -> Tensor:
        #ffn
        b , s , d = input.shape
        assert s==1
        assert d == self.hidden_size
        input = self.ffn_rmsnorm(input) #(b, s, hidden_size)
        up = self.linear_up(input, self.W_linear_up)    #(b, s, intermediate_size)
        gate = self.linear_gate(input, self.W_linear_gate)#(b, s, intermediate_size)
        gate = self.silu(gate)
        swiglu = self.swiglu_mul(up, gate)
        down = self.linear_down(swiglu, self.W_linear_down)#(b, s, hidden_size)
        output = self.ffn_resadd(down, input)

        return output
    
    def mapping_and_simulate(self, device:Device, layer_id: int, micro_batch_id: int, n_layers_per_chip: int):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, duration_cycles: float) -> float:
            weight_bytes = 0
            if op_name == "linear_up":
                weight_bytes = size(self.W_linear_up.shape) * self.datatype.word_size
            elif op_name == "linear_gate":
                weight_bytes = size(self.W_linear_gate.shape) * self.datatype.word_size
            elif op_name == "linear_down":
                weight_bytes = size(self.W_linear_down.shape) * self.datatype.word_size
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
            return float(comp.get('total', 0.0))

        def measure(op_name, op_obj, comm_latency_func=lambda sz: 0, output_shape_override=None, layer_id: int = None, micro_batch_id: int = None):
            output_shape = output_shape_override if output_shape_override is not None else op_obj.output_shape
            output_datasize = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_latency = op_obj.mapping_and_simulate(device)
            communication_latency = comm_latency_func(output_datasize)
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_latency + communication_latency
            operator_latency.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '计算延时': compute_latency, '通信延时': communication_latency, '总延时': total})
            dram_pj = _calc_dram_energy(op_name, total)
            operator_energy.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total, output_datasize

        # ffn_rmsnorm
        t, last_out = measure("ffn_rmsnorm", self.ffn_rmsnorm,
                                lambda sz: reduce_multicast(device) + multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_up
        t, last_out = measure("linear_up", self.linear_up, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_gate
        t, last_out = measure("linear_gate", self.linear_gate, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # silu
        t, last_out = measure("silu", self.silu, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # swiglu_mul
        t, last_out = measure("swiglu_mul", self.swiglu_mul, lambda sz: multicast(device, sz), layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # linear_down
        t, last_out = measure("linear_down", self.linear_down, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        # ffn_resadd
        t, last_out = measure("ffn_resadd", self.ffn_resadd, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t

        if( (layer_id +1) % n_layers_per_chip ==0):
            pipeline_latency = concat(device ,last_out) + p2p(device ,last_out) + scatter(device ,last_out)
            operator_latency.append({'operator': 'pipeline_communication', 'micro_batch_id': micro_batch_id, '总延时': pipeline_latency})
            total_latency += pipeline_latency
        
        return operator_latency, total_latency ,operator_energy



