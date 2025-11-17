from __future__ import annotations
from typing import Dict, Optional
from hardware_model.memory_module import MemoryModule
import math

ACCESS_GRANULARITY_BITS = 256      # 单次访问粒度
ROW_ACCESS_CAPACITY = 128          # 每行可容纳的访问次数(256bit *128 = 32768bit)

class DRAMState:
    """跨算子持久化的DRAM状态, 用于准确统计行激活与刷新次数。

    设计:
    - accesses_in_row: 当前行已经完成的256bit粒度访问次数
    - total_row_activations: 换行产生的激活次数(不含第一行的初始激活)
    - last_refresh_time_ms: 最近一次刷新后的累计时间点
    - total_refreshes: 已刷新次数
    - total_time_ms: 到目前为止的累计运行时间
    注: 初始不计第一行激活, 与"换行才激活"需求一致。
    """
    def __init__(self, memory: MemoryModule, frequency_mhz: float):
        self.memory = memory
        self.frequency_mhz = frequency_mhz
        self.accesses_in_row = 0
        self.total_row_activations = 0
        self.last_refresh_time_ms = 0.0
        self.total_refreshes = 0
        self.total_time_ms = 0.0

    def record(self, read_bytes: float, write_bytes: float, duration_cycles: float) -> Dict[str, float]:
        if self.frequency_mhz <= 0:
            return {"standby":0.0,"activate":0.0,"read":0.0,"write":0.0,"refresh":0.0,"total":0.0}

        op_time_ms = duration_cycles / (self.frequency_mhz * 1e6) * 1e3
        memory = self.memory

        # standby 能耗
        standby_pj = memory.power_ACT_standby * op_time_ms * 1e6

        # 将访问平均分布到各bank, 得到每bank的访问量用于行跨越判断
        eff_read_bytes = math.ceil(max(read_bytes,0.0) / max(getattr(memory,'DRAM_n_bank',1),1))
        eff_write_bytes = math.ceil(max(write_bytes,0.0) / max(getattr(memory,'DRAM_n_bank',1),1))
        read_bits = eff_read_bytes * 8.0
        write_bits = eff_write_bytes * 8.0
        read_accesses = int((read_bits + ACCESS_GRANULARITY_BITS - 1)//ACCESS_GRANULARITY_BITS)
        write_accesses = int((write_bits + ACCESS_GRANULARITY_BITS - 1)//ACCESS_GRANULARITY_BITS)
        new_accesses = read_accesses + write_accesses

        op_row_activations = 0
        remaining = new_accesses
        while remaining > 0:
            capacity_left = ROW_ACCESS_CAPACITY - self.accesses_in_row
            consume = min(remaining, capacity_left)
            self.accesses_in_row += consume
            remaining -= consume
            if self.accesses_in_row == ROW_ACCESS_CAPACITY and remaining > 0:
                # 换到下一行 -> 激活
                self.total_row_activations += 1
                op_row_activations += 1
                self.accesses_in_row = 0

        activate_pj = op_row_activations * memory.ACT_energy_inc * 1e3
        read_pj = read_accesses * memory.RD_energy_inc * 1e3
        write_pj = write_accesses * memory.WR_energy_inc * 1e3

        # 刷新: 按累计时间判断是否跨越刷新周期
        end_time_ms = self.total_time_ms + op_time_ms
        if memory.DRAM_tREF > 0:
            new_refreshes = int((end_time_ms - self.last_refresh_time_ms) // memory.DRAM_tREF)
        else:
            new_refreshes = 0
        if new_refreshes > 0:
            self.total_refreshes += new_refreshes
            self.last_refresh_time_ms += new_refreshes * memory.DRAM_tREF
        refresh_pj = new_refreshes * memory.REF_energy_inc * 1e3

        self.total_time_ms = end_time_ms

        total_pj = standby_pj + activate_pj + read_pj + write_pj + refresh_pj
        return {
            "standby": standby_pj,
            "activate": activate_pj,
            "read": read_pj,
            "write": write_pj,
            "refresh": refresh_pj,
            "total": total_pj,
            "row_activations": op_row_activations,
            "refreshes": new_refreshes,
        }

_GLOBAL_DRAM_STATE: Optional[DRAMState] = None

def reset_global_dram_state(memory: MemoryModule, frequency_mhz: float) -> None:
    global _GLOBAL_DRAM_STATE
    _GLOBAL_DRAM_STATE = DRAMState(memory, frequency_mhz)

def get_global_dram_state(memory: MemoryModule, frequency_mhz: float) -> DRAMState:
    global _GLOBAL_DRAM_STATE
    if _GLOBAL_DRAM_STATE is None:
        _GLOBAL_DRAM_STATE = DRAMState(memory, frequency_mhz)
    return _GLOBAL_DRAM_STATE

def compute_dram_energy(memory: MemoryModule,
                        read_bytes: float,
                        write_bytes: float,
                        duration_cycles: float,
                        device_frequency_mhz: float) -> Dict[str, float]:
    state = get_global_dram_state(memory, device_frequency_mhz)
    return state.record(read_bytes, write_bytes, duration_cycles)

def summarize_dram_components(d: Dict[str, float]) -> float:
    return float(d.get("total", 0.0))
