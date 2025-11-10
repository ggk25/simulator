from utils import data_type_dict
from math import ceil

class VectorUnit:
    def __init__(
        self,
        word_size,
        cycle_per_exp, # cycles per exp instruction
		cycle_per_reciprocal, # cycles per reciprocal instruction
		cycle_per_reciprocal_sqrt, # cycles per reciprocal_square_root instruction
		cycle_per_vector_loop, # cycles per vector loop
        vector_width, # vector width
        data_type=data_type_dict["fp32"],
    ):
        self.word_size = word_size  # Byte
        self.cycle_per_exp = cycle_per_exp  
        self.cycle_per_reciprocal = cycle_per_reciprocal
        self.cycle_per_reciprocal_sqrt = cycle_per_reciprocal_sqrt
        self.cycle_per_vector_loop = cycle_per_vector_loop
        self.vector_width = vector_width
        self.data_type = data_type


vector_unit_dict = {
    "D37x": VectorUnit(4, 15, 12, 15, 2, 32),
}

class SystolicArray:
    def __init__(
        self,
        array_height,
        array_width,
        mac_per_cycle,
        input_word_size,
        output_word_size,
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size


systolic_array_dict = {
    "D37x": SystolicArray(32, 32, 1, 1, 4),
}

class compute_module:
	def __init__(
		self,
		vector_unit:VectorUnit,
		systolic_array:SystolicArray,
	)-> None:
		self.vector_unit = vector_unit
		self.systolic_array = systolic_array

compute_module_dict = {
	"D37x": compute_module(
		vector_unit_dict["D37x"],
		systolic_array_dict["D37x"],
	),
}

		