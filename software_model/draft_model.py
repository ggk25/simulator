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

class DraftModel_Prefill:
    def __init__(self, datatype: DataType, hidden_size=4096 ,concat_size=8192,head_dim=128, \
                num_attention_heads=32, num_key_value_heads=8,intermediate_size= 12288):
        self.datatype = datatype
        self.hidden_size = hidden_size
        self.concat_size = concat_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size

        self.WQ = Tensor([self.concat_size, self.num_attention_heads * self.head_dim], self.datatype)
        self.WK = Tensor([self.concat_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.WV = Tensor([self.concat_size, self.num_key_value_heads * self.head_dim], self.datatype)
        self.WO = Tensor([self.num_attention_heads * self.head_dim, self.hidden_size], self.datatype)
        self.W_linear_up = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_gate = Tensor([self.hidden_size, self.intermediate_size], self.datatype)
        self.W_linear_down = Tensor([self.intermediate_size, self.hidden_size], self.datatype)

    def __call__(self, input: Tensor) -> Tensor:
        b , s , d = input.shape
        assert d == self.hidden_size
        #draft model forward
        
        output =
        return output
    
    def mapping_and_simulate(self, device: Device, layer_id: int = 0, micro_batch_id: int = 0):
        operator_latency = []
        operator_energy = []
        total_latency = 0.0
        counter = get_global_power_counter()
        energy_table = load_energy_table()

        def _calc_dram_energy(op_name: str, duration_cycles: float) -> float:
            weight_bytes = 0
            comp = compute_dram_energy(device.memory, weight_bytes, 0.0, duration_cycles, device.frequency)
            return float(comp.get('total', 0.0))

        GEMM_SET = {}
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