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
    
    def mapping_and_simulate(self, device:Device ):
        operator_latency = []
        total_latency = 0

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
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_up.mapping_and_simulate(device)   #需要选中8个专家
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_gate")
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("silu")
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("swiglu_mul")
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = (self.shared_experts_count+self.selected_expert_count) * multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_down")
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # print("mul")
        output_datasize = size(self.mul.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.mul.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("expert_add")
        output_datasize = size(self.moe_add.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.moe_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "moe_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 

        return operator_latency ,total_latency ,pipeline_latency
    
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
    
    def mapping_and_simulate(self, device:Device ):

        operator_latency = []
        total_latency = 0

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
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_up.mapping_and_simulate(device)   #需要选中8个专家
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_up", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_gate")
        output_datasize = size(self.linear_gate.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_gate.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_gate", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("silu")
        output_datasize = size(self.silu.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.silu.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "silu", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("swiglu_mul")
        output_datasize = size(self.swiglu_mul.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.swiglu_mul.mapping_and_simulate(device)
        communication_latency = (self.shared_experts_count+self.selected_expert_count) * multicast(device ,output_datasize)
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "swiglu_mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("linear_down")
        output_datasize = size(self.linear_down.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.linear_down.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "linear_down", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        
        # print("mul")
        output_datasize = size(self.mul.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.mul.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "mul", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)

        # print("expert_add")
        output_datasize = size(self.moe_add.output_shape) * self.datatype.word_size
        compute_latency = (self.shared_experts_count+self.selected_expert_count) * self.moe_add.mapping_and_simulate(device)
        communication_latency = 0
        total_latency += compute_latency + communication_latency
        operator_latency_dict = {'operator': "moe_add", '计算延时':compute_latency ,\
                                 '通信延时':communication_latency, '总延时':compute_latency + communication_latency}
        operator_latency.append(operator_latency_dict)
        pipeline_latency = concat(device ,output_datasize) +p2p(device ,output_datasize) + scatter(device ,output_datasize) 


        return operator_latency ,total_latency  ,pipeline_latency