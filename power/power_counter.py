from __future__ import annotations
from typing import Dict, Optional, Any
import threading


class PowerCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # counts per operator
        self._ops: Dict[str, Dict[str, float]] = {}
        # totals
        self._totals: Dict[str, float] = {
            "mac_ops_b4": 0.0,
            "mac_ops_b8": 0.0,
            "mac_ops_b16": 0.0,
            "mac_ops_b32": 0.0,
            "eltwise_ops": 0.0,
            "noc_bit_hops": 0.0,
            "pcie_bits": 0.0,
            "dram_bits": 0.0,
            "sram_32b_accesses": 0.0,
        }

    def reset(self) -> None:
        with self._lock:
            self._ops.clear()
            for k in list(self._totals.keys()):
                self._totals[k] = 0.0

    def _ensure_op(self, operator: str) -> Dict[str, float]:
        if operator not in self._ops:
            self._ops[operator] = {
                "mac_ops_b4": 0.0,
                "mac_ops_b8": 0.0,
                "mac_ops_b16": 0.0,
                "mac_ops_b32": 0.0,
                "eltwise_ops": 0.0,
                "noc_bit_hops": 0.0,
                "pcie_bits": 0.0,
                "dram_bits": 0.0,
                "sram_32b_accesses": 0.0,
            }
        return self._ops[operator]

    # recorders
    def add_mac(self, count: float, operator: str, bits: int) -> None:
        key = None
        if bits <= 4:
            key = "mac_ops_b4"
        elif bits <= 8:
            key = "mac_ops_b8"
        elif bits <= 16:
            key = "mac_ops_b16"
        else:
            key = "mac_ops_b32"
        with self._lock:
            op = self._ensure_op(operator)
            op[key] += float(count)
            self._totals[key] += float(count)

    def add_eltwise(self, count: float, operator: str) -> None:
        with self._lock:
            op = self._ensure_op(operator)
            op["eltwise_ops"] += float(count)
            self._totals["eltwise_ops"] += float(count)

    def add_dram_bytes(self, byte_count: float, operator: str) -> None:
        # store in bits to align with energy table (per bit)
        bits = float(byte_count) * 8.0
        with self._lock:
            op = self._ensure_op(operator)
            op["dram_bits"] += bits
            self._totals["dram_bits"] += bits

    def add_pcie_bytes(self, byte_count: float, operator: str = "pcie") -> None:
        bits = float(byte_count) * 8.0
        with self._lock:
            op = self._ensure_op(operator)
            op["pcie_bits"] += bits
            self._totals["pcie_bits"] += bits

    def add_noc_transfer(self, bytes_transferred: float, avg_hops: float, operator: str = "noc") -> None:
        # noc energy modeled as per-bit-per-hop; accumulate bit-hops
        bit_hops = float(bytes_transferred) * 8.0 * float(avg_hops)
        with self._lock:
            op = self._ensure_op(operator)
            op["noc_bit_hops"] += bit_hops
            self._totals["noc_bit_hops"] += bit_hops

    def add_sram_access_32b(self, access_count: float, operator: str) -> None:
        with self._lock:
            op = self._ensure_op(operator)
            op["sram_32b_accesses"] += float(access_count)
            self._totals["sram_32b_accesses"] += float(access_count)

    def get_counts(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {k: v.copy() for k, v in self._ops.items()}

    def get_totals(self) -> Dict[str, float]:
        with self._lock:
            return self._totals.copy()

    def compute_energy(self, energy_table) -> Dict[str, Dict[str, float]]:
        # returns per-operator energy breakdown and totals (pJ)
        counts = self.get_counts()
        results: Dict[str, Dict[str, float]] = {}
        for op, c in counts.items():
            mac_e_4 = c["mac_ops_b4"] * getattr(energy_table, "mac4_pj", 0.0)
            mac_e_8 = c["mac_ops_b8"] * getattr(energy_table, "mac8_pj", 0.0)
            mac_e_16 = c["mac_ops_b16"] * getattr(energy_table, "mac16_pj", 0.0)
            mac_e_32 = c["mac_ops_b32"] * getattr(energy_table, "mac32_pj", 0.0)
            mac_e = mac_e_4 + mac_e_8 + mac_e_16 + mac_e_32
            ew_e = c["eltwise_ops"] * energy_table.eltwise_energy_pj
            noc_e = c["noc_bit_hops"] * energy_table.noc_energy_pj_per_bit_per_hop
            pcie_e = c["pcie_bits"] * energy_table.pcie_energy_pj_per_bit
            dram_e = c["dram_bits"] * energy_table.dram_energy_pj_per_bit
            sram_e = c["sram_32b_accesses"] * getattr(energy_table, "sram_energy_pj_per_32b", 0.0)
            total = mac_e + ew_e + noc_e + pcie_e + dram_e + sram_e
            results[op] = {
                "mac_energy_pj": mac_e,
                "mac4_energy_pj": mac_e_4,
                "mac8_energy_pj": mac_e_8,
                "mac16_energy_pj": mac_e_16,
                "mac32_energy_pj": mac_e_32,
                "eltwise_energy_pj": ew_e,
                "noc_energy_pj": noc_e,
                "pcie_energy_pj": pcie_e,
                "dram_energy_pj": dram_e,
                "sram_energy_pj": sram_e,
                "total_energy_pj": total,
            }
        # totals
        totals = self.get_totals()
        results["__total__"] = {
            "mac_energy_pj": totals["mac_ops_b4"] * getattr(energy_table, "mac4_pj", 0.0)
            + totals["mac_ops_b8"] * getattr(energy_table, "mac8_pj", 0.0)
            + totals["mac_ops_b16"] * getattr(energy_table, "mac16_pj", 0.0)
            + totals["mac_ops_b32"] * getattr(energy_table, "mac32_pj", 0.0),
            "mac4_energy_pj": totals["mac_ops_b4"] * getattr(energy_table, "mac4_pj", 0.0),
            "mac8_energy_pj": totals["mac_ops_b8"] * getattr(energy_table, "mac8_pj", 0.0),
            "mac16_energy_pj": totals["mac_ops_b16"] * getattr(energy_table, "mac16_pj", 0.0),
            "mac32_energy_pj": totals["mac_ops_b32"] * getattr(energy_table, "mac32_pj", 0.0),
            "eltwise_energy_pj": totals["eltwise_ops"] * energy_table.eltwise_energy_pj,
            "noc_energy_pj": totals["noc_bit_hops"] * energy_table.noc_energy_pj_per_bit_per_hop,
            "pcie_energy_pj": totals["pcie_bits"] * energy_table.pcie_energy_pj_per_bit,
            "dram_energy_pj": totals["dram_bits"] * energy_table.dram_energy_pj_per_bit,
            "sram_energy_pj": totals["sram_32b_accesses"] * getattr(energy_table, "sram_energy_pj_per_32b", 0.0),
        }
        t = results["__total__"]
        t["total_energy_pj"] = t["mac_energy_pj"] + t["eltwise_energy_pj"] + t["noc_energy_pj"] + t["pcie_energy_pj"] + t["dram_energy_pj"] + t.get("sram_energy_pj", 0.0)
        return results


_GLOBAL_COUNTER: Optional[PowerCounter] = None


def get_global_power_counter() -> PowerCounter:
    global _GLOBAL_COUNTER
    if _GLOBAL_COUNTER is None:
        _GLOBAL_COUNTER = PowerCounter()
    return _GLOBAL_COUNTER
