import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.device import Device ,device_dict
import math
from .non_gemm import element_wise_mul_add
from power import get_global_power_counter

class matmul:
    def __init__(self, data_type:DataType):
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.data_type = data_type
    
    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M ,self.N, self.K , self.data_type)
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)

        return output
    
    #单核计算的延迟
    def simulate_latency(self, M:int, N:int ,K:int ,data_type:DataType ,device:Device) :
        array_height = device.compute_module.systolic_array.array_height
        array_width = device.compute_module.systolic_array.array_width
        if(data_type == data_type_dict["fp8"]):
            latency = M * (math.ceil(K/array_height)) * (math.ceil(N/array_width))
        elif(data_type == data_type_dict["fp16"]):
            latency = M * (math.ceil(K/(array_height/2))) * (math.ceil(N/array_width))
        elif(data_type == data_type_dict["fp4"]):
            latency = M * (math.ceil(K/(array_height))) * (math.ceil(N/(array_width*2)))
        return latency

    def mapping_and_simulate(self, device:Device):
        array_width = device.compute_module.systolic_array.array_width
        DRAM_bandwidth_per_cycle = device.memory.DRAM_bandwidth* device.memory.DRAM_bandwidth_util * 1e3 / device.frequency
        min_gemm_bubble = math.ceil(self.data_type.word_size * device.compute_module.systolic_array.array_height \
                            * device.compute_module.systolic_array.array_width * device.PE_count * \
                            device.core_count / DRAM_bandwidth_per_cycle)
        
        # record MAC ops, DRAM(B) and SRAM(A,C) traffic for energy
        counter = get_global_power_counter()
        # Total math ops performed logically
        mac_ops = self.M * self.N * self.K
        bits = int(8 * self.data_type.word_size)
        counter.add_mac(mac_ops, operator="matmul", bits=bits)
        # Memory traffic: A from SRAM, B from DRAM, C to SRAM
        # 现在的映射方式是权重按列切分，输入需要复制到每个core上，a的访问量乘上core数
        a_bytes = (self.M * self.K) * self.data_type.word_size * device.core_count
        b_bytes = (self.K * self.N) * self.data_type.word_size
        c_bytes = (self.M * self.N) * self.data_type.word_size
        # SRAM energy modeled per 32-bit access
        sram_access_32b = math.ceil(a_bytes / 4) + math.ceil(c_bytes / 4)
        counter.add_sram_access_32b(sram_access_32b, operator="matmul")
        # DRAM: only B
        counter.add_dram_bytes(b_bytes, operator="matmul")

        if(len(self.input1_shape) > 3):
            M = max(min_gemm_bubble ,self.input1_shape[0] * self.input1_shape[2])
            K = math.ceil(self.input1_shape[-1] / device.PE_count)
            N = math.ceil(self.input2_shape[-1] / device.core_count)
            latency = self.input1_shape[1] * self.simulate_latency(M ,N ,K ,self.data_type ,device) + 2 * array_width
            PE_reduce = element_wise_mul_add(self.data_type)    #PE之间reduce
            reduce = PE_reduce(Tensor([M,N], self.data_type))
            latency += PE_reduce.mapping_and_simulate(device)
        else:
            M = max(min_gemm_bubble ,self.input1_shape[0] * self.input1_shape[1])
            K = math.ceil(self.input1_shape[-1] / device.PE_count)
            N = math.ceil(self.input2_shape[-1] / device.core_count)
            latency = self.simulate_latency(M ,N ,K ,self.data_type, device) + 2 * array_width
            PE_reduce = element_wise_mul_add(self.data_type)
            reduce = PE_reduce(Tensor([M,N], self.data_type))
            latency += PE_reduce.mapping_and_simulate(device)
        return latency
    
class Batched_matmul:
    def __init__(self, data_type:DataType):
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.data_type = data_type

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)

        return output
    
    def mapping_and_simulate(self ,device:Device):
        bs = self.input1_shape[0]
        input1 = Tensor(self.input1_shape[1:] ,self.data_type)
        input2 = Tensor(self.input2_shape[1:] ,self.data_type) 
        '''
        这里虽然是多头，但是MLA和GQA中多个头是共享一个KV的，所以权重复用率会比较高
        ''' 
        batched_matmul = matmul(self.data_type)
        _ = batched_matmul(input1, input2)
        # energy accounting: scale MACs and DRAM by batch size
        counter = get_global_power_counter()
        # for batch, ops already counted inside each inner call, so add only if we bypassed __call__ flow
        batched_matmul_latency = bs * batched_matmul.mapping_and_simulate(device)

        return batched_matmul_latency

'''  
tensor_mul = Batched_matmul(data_type_dict["fp8"])
input1 = Tensor([8, 128, 1 ,64],data_type_dict["fp8"])
input2 = Tensor([8, 128, 64 ,1024],data_type_dict["fp8"])
output = tensor_mul(input1 , input2)
latency = tensor_mul.mapping_and_simulate(device_dict["D37x"])
print(latency)
print(output.shape)
'''
