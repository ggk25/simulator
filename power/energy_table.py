from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import os


@dataclass
class EnergyTable:
    # Energies in picojoules (pJ)
    # MAC energy per precision
    mac4_pj: float = 0.0
    mac8_pj: float = 0.0
    mac16_pj: float = 0.0
    mac32_pj: float = 0.0
    # Element-wise (can be per-op average)
    eltwise_energy_pj: float = 0.0
    # Interconnect and memory
    noc_energy_pj_per_bit_per_hop: float = 0.0
    pcie_energy_pj_per_bit: float = 0.0
    dram_energy_pj_per_bit: float = 0.0
    # SRAM access energy per 32-bit access
    sram_energy_pj_per_32b: float = 0.0
    # SRAM leakage power in Watts (W)
    sram_leakage_power: float = 0.0


def load_energy_table(config_path: Optional[str] = None) -> EnergyTable:
    # Try load from JSON file; if not present, return defaults (zeros)
    if config_path is None:
        # default location: project root power_config.json
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        config_path = os.path.join(here, "power_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EnergyTable(
                mac4_pj=float(data.get("mac4_pj", data.get("mac_energy_pj_4bit", 0.0))),
                mac8_pj=float(data.get("mac8_pj", data.get("mac_energy_pj_8bit", 0.0))),
                mac16_pj=float(data.get("mac16_pj", data.get("mac_energy_pj_16bit", 0.0))),
                mac32_pj=float(data.get("mac32_pj", data.get("mac_energy_pj_32bit", 0.0))),
                eltwise_energy_pj=float(data.get("eltwise_energy_pj", 0.0)),
                noc_energy_pj_per_bit_per_hop=float(data.get("noc_energy_pj_per_bit_per_hop", 0.0)),
                pcie_energy_pj_per_bit=float(data.get("pcie_energy_pj_per_bit", 0.0)),
                dram_energy_pj_per_bit=float(data.get("dram_energy_pj_per_bit", 0.0)),
                sram_energy_pj_per_32b=float(data.get("sram_energy_pj_per_32b", 0.0)),
                sram_leakage_power=float(data.get("sram_leakage_power", 0.0)),
            )
        except Exception:
            # fall back to defaults if parsing failed
            return EnergyTable()
    return EnergyTable()
