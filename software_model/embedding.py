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

class Embedding:
    def __init__(self, device: Device, vocab_size: int, hidden_size: int, datatype):
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.datatype = datatype

        self.W_embedding = Tensor([self.vocab_size, self.hidden_size], self.datatype)
        self.embedding = matmul(self.datatype)

    def __call__(self, input: Tensor)-> Tensor:
        b, s, d = input.shape
        assert d == self.vocab_size #Input hidden size must match embedding dimension
        output = self.embedding(input, self.W_embedding)
        return output
    
    def mapping_and_simulate(self, device: Device, layer_id: int = 0, micro_batch_id: int = 0):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, duration_cycles: float) -> float:
            weight_bytes = 0
            if op_name == "Embedding":
                weight_bytes = size(self.W_embedding.shape) * self.datatype.word_size
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
            return float(comp.get('total', 0.0))

        GEMM_SET = {"Embedding"}
        ELEMENTWISE_SET = {}

        def measure(op_name, op_obj, comm_latency_func=lambda sz: 0, output_shape_override=None, layer_id: int = None, micro_batch_id: int = None):
            output_shape = output_shape_override if output_shape_override is not None else op_obj.output_shape
            output_datasize = size(output_shape) * self.datatype.word_size
            energy_before = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            compute_latency = op_obj.mapping_and_simulate(device)
            comm_val = comm_latency_func(output_datasize)
            if isinstance(comm_val, tuple):
                onchip_comm_latency, pcie_comm_latency = comm_val
            else:
                onchip_comm_latency = comm_val
                pcie_comm_latency = 0.0
            communication_latency = onchip_comm_latency + pcie_comm_latency
            energy_after = counter.compute_energy(energy_table).get("__total__", {}).get("total_energy_pj", 0.0)
            energy_delta = energy_after - energy_before
            total = compute_latency + communication_latency
            gemm_latency = compute_latency if op_name in GEMM_SET else 0.0
            eltwise_latency = compute_latency if op_name in ELEMENTWISE_SET else 0.0
            operator_latency.append({
                'operator': op_name,
                'layer_id': layer_id,
                'micro_batch_id': micro_batch_id,
                '计算延时': compute_latency,
                '通信延时': communication_latency,
                '总延时': total,
                'GEMM延时': gemm_latency,
                'ElementWise延时': eltwise_latency,
                '片上通信延时': onchip_comm_latency,
                'PCIe延时': pcie_comm_latency
            })
            dram_pj = _calc_dram_energy(op_name, total)
            operator_energy.append({'operator': op_name, 'layer_id': layer_id, 'micro_batch_id': micro_batch_id, '总延时': total, 'logic能耗': energy_delta, 'DRAM能耗': dram_pj})
            return total, output_datasize

        t, last_out = measure("Embedding", self.embedding, layer_id=layer_id, micro_batch_id=micro_batch_id)
        total_latency += t
        
        return operator_latency, total_latency, operator_energy
        