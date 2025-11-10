import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.device import Device ,device_dict
from power import get_global_power_counter
import math

class softmax:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self, M:int ,N:int ,device:Device):
        cycle_per_exp = device.compute_module.vector_unit.cycle_per_exp
        cycle_per_vector_loop = device.compute_module.vector_unit.cycle_per_vector_loop
        cycle_per_reciprocal = device.compute_module.vector_unit.cycle_per_reciprocal
        vector_unit_width = device.compute_module.vector_unit.vector_width

        #PE内寻找最大值
        find_max_delay = cycle_per_vector_loop + vector_unit_width + M * math.ceil(N / vector_unit_width)
        #减最大值、计算exp函数并向后累加
        exp_accumulation_delay = vector_unit_width + cycle_per_vector_loop + \
                                (1 + cycle_per_exp + 1) * M * math.ceil(N / vector_unit_width)
        #计算指数和的倒数
        reduce_sum_delay = cycle_per_reciprocal
        #逐元素乘
        elementwise_mul_delay = vector_unit_width + M * math.ceil(N / vector_unit_width)

        return find_max_delay + exp_accumulation_delay + \
                reduce_sum_delay + elementwise_mul_delay
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        # energy accounting，4 bytes for 1 element
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # Approximate 14 element-wise ops per element:(reduce_max + sub + exp(10) + reduce_sum + mul)
        counter.add_eltwise(14 * total_elems, operator="softmax")
        return self.simulate_latency(M ,N ,device)
    
class rmsnorm:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,M:int ,N:int ,device:Device):
        cycle_per_vector_loop = device.compute_module.vector_unit.cycle_per_vector_loop
        cycle_per_reciprocal_sqrt = device.compute_module.vector_unit.cycle_per_reciprocal_sqrt
        vector_unit_width = device.compute_module.vector_unit.vector_width

         #PE上计算平方和
        square_accumulation_delay = cycle_per_vector_loop + vector_unit_width + math.ceil((M * N)/vector_unit_width)
        #计算平方根倒数
        reduce_sum_delay = cycle_per_reciprocal_sqrt
        #逐元素乘
        elementwise_mul_delay = vector_unit_width + math.ceil((M * N)/vector_unit_width)
        return square_accumulation_delay + reduce_sum_delay + elementwise_mul_delay
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # rmsnorm ~ 3 ops per element (square + reduce_sum + mul)
        counter.add_eltwise(3 * total_elems, operator="rmsnorm")
        return self.simulate_latency(M ,N ,device)
    
class rope:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,M:int ,N:int ,device:Device):
        vector_unit_width = device.compute_module.vector_unit.vector_width
        latency = vector_unit_width + 3 * math.ceil((M * N) / vector_unit_width)

        return latency
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # rope ~ 3 ops per element (sin/cos pair usage)
        counter.add_eltwise(3 * total_elems, operator="rope")
        return self.simulate_latency(M ,N ,device)
    
class silu:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,M:int ,N:int ,device:Device):
        cycle_per_exp = device.compute_module.vector_unit.cycle_per_exp
        cycle_per_reciprocal = device.compute_module.vector_unit.cycle_per_reciprocal
        vector_unit_width = device.compute_module.vector_unit.vector_width

        latency = vector_unit_width + (cycle_per_exp + 1 + cycle_per_reciprocal + 1)\
                    * math.ceil((M * N) / vector_unit_width)

        return latency
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # silu ~ 17 ops per element (sigmoid(17) + mul)
        counter.add_eltwise(17 * total_elems, operator="silu")
        return self.simulate_latency(M ,N ,device)
    
class sigmoid:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,M:int ,N:int ,device:Device):
        cycle_per_exp = device.compute_module.vector_unit.cycle_per_exp
        cycle_per_reciprocal = device.compute_module.vector_unit.cycle_per_reciprocal
        vector_unit_width = device.compute_module.vector_unit.vector_width

        latency = vector_unit_width + (cycle_per_exp + 1 + cycle_per_reciprocal)\
                    * math.ceil((M * N) / vector_unit_width)
        return latency
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # sigmoid ~ 16 ops per element (exp(10) + reciprocal(6))
        counter.add_eltwise(16 * total_elems, operator="sigmoid")
        return self.simulate_latency(M ,N ,device)
        
class element_wise_mul_add:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input1: Tensor ,input2: Tensor=Tensor([1,1],data_type_dict["fp16"])) -> Tensor:
        self.input_shape = input1.shape
        self.output_shape = input1.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,M:int ,N:int ,device:Device):
        vector_unit_width = device.compute_module.vector_unit.vector_width

        latency = vector_unit_width + math.ceil((M * N) / vector_unit_width)
        return latency
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = math.ceil(self.input_shape[-1] / (device.core_count*device.PE_count))
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        # element-wise mul/add: count as 1 op per element by default
        counter.add_eltwise(1 * total_elems, operator="element_wise_mul_add")
        return self.simulate_latency(M ,N ,device)
    
class rank:
    def __init__(self, data_type:DataType):
        self.input_shape = None
        self.output_shape = None
        self.data_type = data_type
        
    class ComputationalGraph:
        def __init__(self, M: int, N: int , data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N},  word_size(B): {self.data_type.word_size}"
            )

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        self.output_shape = input.shape
        self.M = size(self.input_shape[:-1])
        self.N = self.input_shape[-1]
        self.computationalGraph = self.ComputationalGraph(self.M , self.N , self.data_type)
        output = Tensor(self.output_shape, self.data_type)
        return output
    
    def simulate_latency(self ,N:int ,device:Device):
        latency = 2 * N
        return latency
    
    def mapping_and_simulate(self, device:Device):
        M = size(self.input_shape[:-1])
        N = self.input_shape[-1]
        # Ranking energy is minor; treat comparisons as element-wise ~ logN factors are ignored here
        counter = get_global_power_counter()
        total_elems = size(self.input_shape)
        counter.add_eltwise(1 * total_elems, operator="rank")
        return self.simulate_latency(N ,device)
'''
rmsnorm_test = rmsnorm(data_type_dict["fp8"])
input = Tensor([8192 ,7168],data_type_dict["fp8"])
output = rmsnorm_test(input)
latency = rmsnorm_test.mapping_and_simulate(device_dict["D37x"])
print(output.shape)
print(latency)
'''