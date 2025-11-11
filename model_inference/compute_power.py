import math
from typing import Dict, List, Tuple, Optional
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from power import get_global_power_counter
from power.energy_table import load_energy_table
from utils import Tensor, data_type_dict
from hardware_model.device import device_dict
from model_inference.inference import FlexibleModel, ModelConfig


def _cycles_to_ms(cycles: float, freq_mhz: float) -> float:
    # device.frequency is MHz based on existing code
    return float(cycles) / (freq_mhz * 1e6) * 1e3


def _build_single_chip_timeline(
    model_cfg: ModelConfig,
    device,
    batch_size: int,
    prefill_length: int,
    decode_length: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Build a single-chip execution timeline (ordered segments) for one full run:
    - Repeats per-layer work for ceil(layers/chips) layers on this chip
    - Prefill once per layer, then Decode for each token
    Returns:
      segments: list of {"stage","op","duration_ms"}
      op_total_time_ms: mapping of operator -> total active time (ms) on this chip
    Note: a synthetic op "__pipeline__" is inserted to account for pipeline bubbles if reported.
    """
    micro_batch = batch_size / device.n_chip
    freq = device.frequency

    segments: List[Dict] = []
    op_total_time_ms: Dict[str, float] = {}

    layers_per_chip = math.ceil(model_cfg.layer_count / device.n_chip)

    # helper to accumulate
    def add_segment(stage: str, op: str, cycles: float):
        ms = _cycles_to_ms(cycles, freq)
        segments.append({"stage": stage, "op": op, "duration_ms": ms})
        op_total_time_ms[op] = op_total_time_ms.get(op, 0.0) + ms

    # Build timeline and, at the same time, record power counters by executing mapping
    counter = get_global_power_counter()
    counter.reset()
    energy_table = load_energy_table()

    # datatype already resolved in model_cfg
    dt = model_cfg.datatype

    # Per-layer timeline
    for _layer in range(layers_per_chip):
        # Prefill (single layer on this chip)
        prompt = Tensor([micro_batch, prefill_length, model_cfg.hidden_size], data_type=dt)
        prefill_model = FlexibleModel(model_cfg, stage="prefill")
        _ = prefill_model(prompt)
        pre_ops, pre_total, pre_pipeline = prefill_model.mapping_and_simulate(device)
        for op in pre_ops:
            add_segment("prefill", op["operator"], op["总延时"])
        if pre_pipeline and pre_pipeline > 0:
            add_segment("prefill", "__pipeline__", pre_pipeline)

        # Decode tokens (single layer) with increasing context
        for i in range(1, decode_length + 1):
            ctx = prefill_length + i
            inp = Tensor([micro_batch, 1, model_cfg.hidden_size], data_type=dt)
            decode_model = FlexibleModel(model_cfg, stage="decode", context_length=ctx)
            _ = decode_model(inp)
            dec_ops, dec_total, dec_pipeline = decode_model.mapping_and_simulate(device)
            for op in dec_ops:
                add_segment("decode", op["operator"], op["总延时"])
            if dec_pipeline and dec_pipeline > 0:
                add_segment("decode", "__pipeline__", dec_pipeline)

    # Compute energies for this single-chip run
    energy_results = counter.compute_energy(energy_table)

    return segments, op_total_time_ms, energy_results


def compute_operator_average_power(
    model_cfg: ModelConfig,
    device_name: str,
    batch_size: int,
    prefill_length: int,
    decode_length: int,
) -> pd.DataFrame:
    """
    Compute per-operator average power (W) during active execution time.
    - Builds a single-chip, full-run timeline (ceil(layers/chips) layers).
    - Uses recorded energy per operator from power counters.
    - Average power per operator = total_energy(op) / total_active_time(op).
    The value is per chip and equals the system per-operator average when scaled properly.
    """
    device = device_dict.get(device_name, list(device_dict.values())[0])

    segments, op_time_ms, energy_results = _build_single_chip_timeline(
        model_cfg, device, batch_size, prefill_length, decode_length
    )

    rows = []
    for op, t_ms in op_time_ms.items():
        if op == "__pipeline__" or t_ms <= 0:
            continue
        er = energy_results.get(op, None)
        if not er:
            # No energy accounted (e.g., synthetic op); skip
            continue
        total_pj = float(er.get("total_energy_pj", 0.0))
        avg_w = total_pj / t_ms * 1e-9
        rows.append({
            "算子": op,
            "总能耗(pJ)": total_pj,
            "总时长(ms)": t_ms,
            "平均功耗(W)": avg_w,
        })

    df = pd.DataFrame(rows).sort_values(by="平均功耗(W)", ascending=False)
    return df


def compute_time_window_power(
    model_cfg: ModelConfig,
    device_name: str,
    batch_size: int,
    prefill_length: int,
    decode_length: int,
    window_ms: float = 1.0,
) -> pd.DataFrame:
    """
    Compute approximate real-time power with a fixed window (ms):
    1) Build a single-chip, ordered timeline (ceil(layers/chips) layers).
    2) Obtain per-operator total energy from power counters for this exact run.
    3) Distribute each operator's energy across its segments proportional to time.
    4) Accumulate per-window energy; finally scale by n_chip to approximate system power.

    Returns DataFrame with [t_start_ms, t_end_ms, avg_power_W].
    """
    device = device_dict.get(device_name, list(device_dict.values())[0])

    segments, op_time_ms, energy_results = _build_single_chip_timeline(
        model_cfg, device, batch_size, prefill_length, decode_length
    )

    # Compute per-operator energy (pJ) for this single-chip run
    op_energy_pj: Dict[str, float] = {}
    for op, er in energy_results.items():
        if op == "__total__":
            continue
        op_energy_pj[op] = float(er.get("total_energy_pj", 0.0))

    # Assign energy to each segment proportional to its duration
    seg_energy_pj: List[float] = []
    for seg in segments:
        op = seg["op"]
        dur_ms = seg["duration_ms"]
        if op == "__pipeline__" or dur_ms <= 0:
            seg_energy_pj.append(0.0)
            continue
        total_time = op_time_ms.get(op, 0.0)
        total_energy = op_energy_pj.get(op, 0.0)
        e = total_energy * (dur_ms / total_time) if total_time > 0 else 0.0
        seg_energy_pj.append(e)

    # Build windows for single chip
    total_time_ms = sum(seg["duration_ms"] for seg in segments)
    if window_ms <= 0:
        window_ms = 1.0
    n_windows = int(math.ceil(total_time_ms / window_ms))

    # Accumulate single-chip window energies
    window_energy_pj = [0.0 for _ in range(n_windows)]

    # Sweep segments onto windows
    t_cursor = 0.0
    for seg, e_pj in zip(segments, seg_energy_pj):
        seg_start = t_cursor
        seg_end = t_cursor + seg["duration_ms"]
        # distribute segment energy linearly over its duration
        if seg_end <= seg_start:
            t_cursor = seg_end
            continue
        remaining = seg["duration_ms"]
        cur = seg_start
        while remaining > 0:
            win_idx = int(cur // window_ms)
            win_start = win_idx * window_ms
            win_end = min((win_idx + 1) * window_ms, total_time_ms)
            take = min(win_end - cur, remaining)
            frac = take / seg["duration_ms"]
            window_energy_pj[win_idx] += e_pj * frac
            remaining -= take
            cur += take
        t_cursor = seg_end

    # Scale by number of chips to approximate system-level power
    scale = device.n_chip
    rows = []
    for i in range(n_windows):
        e_pj = window_energy_pj[i] * scale
        t0 = i * window_ms
        t1 = min((i + 1) * window_ms, total_time_ms)
        width = max(t1 - t0, 1e-9)
        avg_w = e_pj / width * 1e-9
        rows.append({
            "t_start_ms": t0,
            "t_end_ms": t1,
            "avg_power_W": avg_w,
        })

    return pd.DataFrame(rows)


# Convenience runner for quick manual checks
if __name__ == "__main__":
    # Example: use the first config style from model_config.json defaults
    # Users can import these functions and pass their own ModelConfig.
    example_cfg = ModelConfig(
        name="seedoss",
        attention_type="GQA",
        ffn_type="FFN",
        hidden_size=5120,
        head_dim=128,
        num_attention_heads=80,
        num_key_value_heads=8,
        ffn_intermediate_size=27648,
        layer_count=64,
        datatype=data_type_dict.get("fp8"),
    )
    device_name = "D37x"
    batch = 256
    prefill = 1024
    decode = 128  # keep small for quick check

    op_df = compute_operator_average_power(example_cfg, device_name, batch, prefill, decode)
    ts_df = compute_time_window_power(example_cfg, device_name, batch, prefill, decode, window_ms=1.0)

    print("Per-operator average power (top 5):")
    print(op_df.head())
    print("\nTimeseries power (first 10 windows):")
    print(ts_df.head(10))
