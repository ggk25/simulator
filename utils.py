from typing import List

def size_of_list(list: List):
    result = 1
    for i in list:
        result *= i
    return result

def size(list):
    # 允许 None：当算子未显式提供 output_shape 时返回 0
    if list is None:
        return 0
    if isinstance(list, List):     
        return size_of_list(list)
    else:
        return getattr(list, 'size', 0)

def closest_factors(n):
    x = int(n**0.5)
    while x >= 1:
        if n % x == 0:
            return x, n // x
        x -= 1
    return 0,0
	
class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        self.word_size:int = word_size

data_type_dict = {"int8": DataType("int8", 1), "fp8": DataType("fp8", 1),"fp4": DataType("fp4", 0.5),
				"fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4)}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type



