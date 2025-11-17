
import json
import os

class MemoryModule:
    def __init__(
        self,
        DRAM_bandwidth,	# GB/s
		DRAM_bandwidth_util,	# %
        DRAM_freq,  # MHz
        DRAM_n_bank,
        DRAM_n_TSV_per_bank,
        DRAM_n_row_per_bank,
        DRAM_n_col_per_bank,
        burst_cycle,
        #DRAM时序参数，
        DRAM_tRAS,  #ns
        DRAM_tRP,   #ns
        DRAM_tRC,   #ns
        DRAM_tREF,  #ms
        #电压域
        VDD1,
        VDD2,
        #DRAM电流参数，单位mA，归一化到per bank
        IDD2_1, #All bank idle
        IDD2_2,
        IDD4WA_1,   #All bank write
        IDD4WA_2,
        IDD4RA_1,   #All bank read
        IDD4RA_2,
        IDD0A_1,    #All bank pre-act
        IDD0A_2,
        IDD3A_1,    #All bank act stand by
        IDD3A_2,
    ):
        self.DRAM_bandwidth = DRAM_bandwidth
        self.DRAM_bandwidth_util = DRAM_bandwidth_util
        self.DRAM_freq = DRAM_freq
        self.burst_cycle = burst_cycle / DRAM_freq * 1e3  # ns
        self.DRAM_n_row_per_bank = DRAM_n_row_per_bank
        self.DRAM_n_col_per_bank = DRAM_n_col_per_bank
        self.DRAM_n_bank = DRAM_n_bank
        self.DRAM_n_TSV_per_bank = DRAM_n_TSV_per_bank
        self.DRAM_tRAS = DRAM_tRAS  # ns
        self.DRAM_tRP = DRAM_tRP    # ns
        self.DRAM_tRC = DRAM_tRC    # ns
        self.DRAM_tREF = DRAM_tREF  # ms
        # active energy increment
        self.ACT_energy_inc = DRAM_n_bank*(VDD1*(IDD0A_1*DRAM_tRC - IDD3A_1*DRAM_tRAS - IDD2_1*DRAM_tRP)\
                        +VDD2*(IDD0A_2*DRAM_tRC - IDD3A_2*DRAM_tRAS - IDD2_2*DRAM_tRP))/1e3  # nJ
        self.RD_energy_inc = DRAM_n_bank*((VDD1*(IDD4RA_1-IDD3A_1)+VDD2*(IDD4RA_2-IDD3A_2))*self.burst_cycle)/1e3  # nJ
        self.WR_energy_inc = DRAM_n_bank*((VDD1*(IDD4WA_1-IDD3A_1)+VDD2*(IDD4WA_2-IDD3A_2))*self.burst_cycle)/1e3  # nJ
        self.REF_energy_inc = self.ACT_energy_inc * self.DRAM_n_row_per_bank  # nJ
        # standby power
        self.power_IDLE = DRAM_n_bank*(VDD1*IDD2_1 + VDD2*IDD2_2)   # mW
        self.power_ACT_standby = DRAM_n_bank*(VDD1*IDD3A_1 + VDD2*IDD3A_2)    # mW

    @classmethod
    def from_dict(cls, params: dict):
        """Create a MemoryModule from a parameter dict.

        The dict keys must match the __init__ parameter names exactly.
        """
        return cls(**params)

    @classmethod
    def from_json(cls, filepath: str):
        #Load parameters from a JSON file and create a MemoryModule.

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Memory config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            params = json.load(f)
        if not isinstance(params, dict):
            raise ValueError("Memory config JSON must be an object/dictionary of parameters")
        return cls.from_dict(params)
    
# Attempt to load a default `memory_config.json` located next to this file.
# Do not raise on import; if the file is missing, leave `memory_dict` empty.
memory_dict = {}
_default_path = os.path.join(os.path.dirname(__file__), 'memory_config.json')
if os.path.isfile(_default_path):
    try:
        memory = MemoryModule.from_json(_default_path)
        memory_dict['D37x'] = memory
    except Exception as e:
        # Warn but don't raise during import.
        print(f"Warning: failed to load memory config '{_default_path}': {e}")
# Example usage:
# from hardware_model.memory_module import MemoryModule
# m = MemoryModule.from_json('path/to/memory_config.json')
# memory_dict['D37x'] = m
