
class MemoryModule:
    def __init__(
        self,
        DRAM_bandwidth,	# GB/s
		DRAM_bandwidth_util,	# %
    ):
        self.DRAM_bandwidth = DRAM_bandwidth
        self.DRAM_bandwidth_util = DRAM_bandwidth_util

memory_dict = {
    "D37x": MemoryModule(12288, 0.8),
}
