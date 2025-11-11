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
    
    def mapping_and_simulate(self, device:Device):
        #attention
        operator_latency = []
        operator_energy = []
        total_latency = 0
        counter = get_global_power_counter()
        energy_table = load_energy_table()
        # ffn_rmsnorm
        output_datasize = size(self.ffn_rmsnorm.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.ffn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_rmsnorm", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "ffn_rmsnorm", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # linear_up
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_up.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_up", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # linear_gate
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_gate", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # silu
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.silu.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "silu", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # swiglu_mul
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "swiglu_mul", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # linear_down
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_down", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # ffn_resadd
        output_datasize = size(self.ffn_resadd.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.ffn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_resadd", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "ffn_resadd", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 
        self.operator_energy = operator_energy
        return operator_latency, total_latency, pipeline_latency
    
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
        operator_energy = []
        total_latency = 0
        counter = get_global_power_counter()
        energy_table = load_energy_table()
        # ffn_rmsnorm
        output_datasize = size(self.ffn_rmsnorm.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.ffn_rmsnorm.mapping_and_simulate(device)
        communication_latency = reduce_multicast(device) + multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_rmsnorm", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "ffn_rmsnorm", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        
        # linear_up
        output_datasize = size(self.linear_up.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_up.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_up", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # linear_gate
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_gate", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # silu
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.silu.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "silu", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # swiglu_mul
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = multicast(device, output_datasize)
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "swiglu_mul", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # linear_down
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "linear_down", '总延时': compute_latency + communication_latency, '能耗': energy_delta})

        # ffn_resadd
        output_datasize = size(self.ffn_resadd.output_shape) * self.datatype.word_size
        energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        compute_latency = self.ffn_resadd.mapping_and_simulate(device)
        communication_latency = 0
        energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
        energy_delta = energy_after - energy_before
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "ffn_resadd", '计算延时':compute_latency ,\
                                    '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        operator_energy.append({'operator': "ffn_resadd", '总延时': compute_latency + communication_latency, '能耗': energy_delta})
        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 
        self.operator_energy = operator_energy
        return operator_latency, total_latency ,pipeline_latency



