from .compute_module import compute_module_dict,compute_module
from .memory_module import memory_dict,MemoryModule
from .communication import communication_dict,communication

class Device:
	def __init__(
		self,
		compute_module:compute_module,
		memory:MemoryModule,
		communication:communication,
		PE_count:int,	#每个核中PE的数量
		core_count:int,	#核数
		row_core_count:int,	#每行核数
		col_core_count:int,	#每列核数
		frequency:int,	#频率：MHz
		n_chip:int
	)-> None:
		self.compute_module = compute_module
		self.memory = memory
		self.communication = communication
		self.PE_count = PE_count
		self.core_count = core_count
		self.row_core_count = row_core_count
		self.col_core_count = col_core_count
		self.frequency = frequency
		self.n_chip = n_chip

device_dict = {
	"D37x": Device(
		compute_module_dict["D37x"],
		memory_dict["D37x"],
		communication_dict["D37x"],
		4,
		15,
		4,
		4,
		800,
		1
	),
}
