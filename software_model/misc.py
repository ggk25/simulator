import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.device import Device
from typing import List ,Tuple
import math

class quantization:
    def __init__(self, source_datatype: DataType ,des_datatype: DataType):
        self.source_datatype = source_datatype
        self.des_datatype = des_datatype

    def __call__(self, tensor:Tensor):
        input = tensor
        output = Tensor(input.shape , self.des_datatype)

        return output

class reshape():
    def __init__(self, data_type: DataType):
        self.data_type = data_type
        self.input_shape = None
        self.output_shape = None

    def __call__(self, input: Tensor, output_shape: List[int]) -> Tensor:
        assert input.size == size(output_shape)
        self.input_shape = input.shape
        self.output_shape = output_shape
        output = Tensor(output_shape, self.data_type)
        return output
    
class split():
    def __init__(self, data_type: DataType):
        self.datatype = data_type
        self.input_shape = None
        self.output_shape = None
    def __call__(self, input: Tensor, split_size: List[int], dim) -> Tuple['Tensor', ...]:
        output = []
        output_shape = List[List]
        assert sum(split_size) == input.shape[dim]

        for sz in split_size:
            new_shape = list(input.shape)
            new_shape[dim] = sz
            output.append(Tensor(new_shape ,self.datatype))
        return tuple(output)
    
class transpose():
    def __init__(self, data_type: DataType):
        self.data_type = data_type
        self.input_shape = None
        self.output_shape = None

    def __call__(self, input: Tensor, permute: List[int]) -> Tensor:
        assert len(input.shape) == len(permute)
        self.input_shape = input.shape
        self.permute = permute

        self.output_shape = [self.input_shape[i] for i in permute]
        output = Tensor(self.output_shape, self.data_type)
        return output

class Concat:
    def __init__(self, data_type: DataType):
        self.datatype = data_type
        self.input_shape  = None   # 仅记录，可留空
        self.output_shape = None   # 仅记录，可留空

    def __call__(self, inputs: Tuple[Tensor, ...], dim: int) -> Tensor:
        """
        沿 dim 维拼接多个 Tensor,返回新的 Tensor。
        要求:inputs 中所有 Tensor 的维度数一致，且除 dim 维外其余维长度均相等。
        """
        if not inputs:
            raise ValueError("Concat needs at least one Tensor.")

        ref_shape = list(inputs[0].shape)
        ndims = len(ref_shape)

        if not (0 <= dim < ndims):
            raise IndexError(f"dim={dim} out of range for {ndims}-D tensor.")

        # 计算拼接后的 shape
        concat_size = 0
        for t in inputs:
            if list(t.shape[:dim]) + list(t.shape[dim+1:]) != \
               ref_shape[:dim] + ref_shape[dim+1:]:
                raise ValueError("All tensors must have the same shape except the concat dimension.")
            concat_size += t.shape[dim]

        new_shape = ref_shape[:]
        new_shape[dim] = concat_size

        return Tensor(new_shape, self.datatype)
'''
tensor = Tensor([8,128,64,32] ,data_type=data_type_dict["fp16"])
transpose_test = transpose(data_type=data_type_dict["fp16"])
tensor = transpose_test(tensor,[3,2,0,1])
print(tensor.shape)
'''
    
