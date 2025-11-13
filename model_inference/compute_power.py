"""
时间窗口功耗计算模块

设计思路：
1. Pipeline并行模型：
   - 32个芯片按流水线方式处理模型层
   - 每个芯片负责 ceil(64层/32芯片) = 2层
   - Batch被切分成多个micro-batch在流水线中流动
   
2. 单芯片功耗分析：
   - 只计算第一个芯片（chip0）的执行时间线和功耗
   - chip0会连续处理多个micro-batch的多层
   - 所有芯片的功耗特性相同，因此单芯片分析即可代表整体
   
3. 时间窗口功耗计算原理：
   - 设定固定时间窗口（如2ms）
   - 将单芯片的所有算子按执行时间线排列
   - 对于跨窗口的算子，假设功耗均匀分布，按时间比例分配能耗
   - 窗口功耗 = 窗口内总能耗 / 窗口时长
   
4. 实现步骤：
   a) 从inference.py收集的算子能耗和延时数据构建单芯片时间线
   b) 根据层数、micro-batch数扩展到完整执行流程（仅chip0）
   c) 按时间窗口切分，计算每个窗口内的平均功耗
"""

from typing import List, Dict, Tuple
import math


class OperatorExecution:
    """单个算子的执行信息"""
    def __init__(self, name: str, start_time_ms: float, duration_ms: float, 
                 energy_pj: float, chip_id: int = 0, layer_id: int = 0, 
                 microbatch_id: int = 0):
        self.name = name
        self.start_time_ms = start_time_ms  # 开始时间（毫秒）
        self.duration_ms = duration_ms       # 持续时间（毫秒）
        self.energy_pj = energy_pj          # 总能耗（皮焦）
        self.chip_id = chip_id              # 芯片ID
        self.layer_id = layer_id            # 层ID
        self.microbatch_id = microbatch_id  # micro-batch ID
        
    @property
    def end_time_ms(self) -> float:
        """结束时间"""
        return self.start_time_ms + self.duration_ms
    
    @property
    def avg_power_w(self) -> float:
        """平均功耗（瓦特）
        
        计算公式：
        功耗(W) = 能耗(pJ) / 时间(ms) × 10^-9
                = 能耗(10^-12 J) / 时间(10^-3 s) × 10^-9
                = 能耗 × 10^-12 / (时间 × 10^-3)
                = 能耗 / 时间 × 10^-9
        """
        if self.duration_ms > 0:
            return self.energy_pj / self.duration_ms * 1e-9
        return 0.0


class TimeWindow:
    """时间窗口"""
    def __init__(self, start_ms: float, duration_ms: float):
        self.start_ms = start_ms
        self.duration_ms = duration_ms
        self.end_ms = start_ms + duration_ms
        self.total_energy_pj = 0.0  # 窗口内总能耗
        self.operators: List[Tuple[OperatorExecution, float]] = []  # (算子, 在窗口内的能耗)
        
    def add_operator(self, op: OperatorExecution) -> None:
        """将算子添加到窗口，计算该算子在窗口内贡献的能耗"""
        # 计算算子与窗口的重叠区间
        overlap_start = max(op.start_time_ms, self.start_ms)
        overlap_end = min(op.end_time_ms, self.end_ms)
        
        if overlap_end <= overlap_start:
            return  # 无重叠
        
        overlap_duration = overlap_end - overlap_start
        
        # 假设算子功耗均匀分布，按时间比例分配能耗
        if op.duration_ms > 0:
            energy_in_window = op.energy_pj * (overlap_duration / op.duration_ms)
        else:
            energy_in_window = 0.0
            
        self.total_energy_pj += energy_in_window
        self.operators.append((op, energy_in_window))
    
    @property
    def avg_power_w(self) -> float:
        """窗口平均功耗（瓦特）"""
        if self.duration_ms > 0:
            return self.total_energy_pj / self.duration_ms * 1e-9
        return 0.0


def build_operator_timeline(
    operator_data: List[Dict],
    layers_per_chip: int,
    num_microbatches: int,
    num_chips: int,
    leakage_power_w: float = 0.0
) -> List[OperatorExecution]:
    """
    构建单个芯片的算子执行时间线
    
    注意：只计算第一个芯片(chip0)的执行时间线，用于分析单芯片功耗特性
    
    Args:
        operator_data: 单个micro-batch单层的算子数据列表，每项包含:
            - 'operator': 算子名称
            - '总延时(ms)': 延时
            - '能耗(pJ)': 能耗
        layers_per_chip: 每个芯片处理的层数
        num_microbatches: micro-batch数量
        num_chips: 芯片数量（用于计算micro-batch数量，但只模拟chip0）
        leakage_power_w: 静态泄漏功耗（瓦特）
        
    Returns:
        按时间排序的算子执行列表（仅chip0）
    """
    timeline: List[OperatorExecution] = []
    
    # 单层所有算子的总时间（单个micro-batch在单个芯片上）
    single_layer_time_ms = sum(op.get('总延时(ms)', 0.0) for op in operator_data)
    
    if single_layer_time_ms <= 0:
        return timeline
    
    # 只计算第一个芯片（chip0）的时间线
    chip_id = 0
    current_time = 0.0
    
    # 为每个micro-batch生成执行序列（chip0会处理所有micro-batch）
    for mb_id in range(num_microbatches):
        # 在当前芯片上处理layers_per_chip层
        for layer_idx in range(layers_per_chip):
            # 为每个算子创建执行记录
            for op_data in operator_data:
                op_name = op_data.get('算子', op_data.get('operator', 'unknown'))
                op_duration = op_data.get('总延时(ms)', 0.0)
                op_energy = op_data.get('能耗(pJ)', 0.0)
                
                # 添加泄漏功耗贡献
                leakage_energy_pj = leakage_power_w * op_duration * 1e9  # W * ms = 1e9 pJ
                total_energy = op_energy + leakage_energy_pj
                
                op_exec = OperatorExecution(
                    name=op_name,
                    start_time_ms=current_time,
                    duration_ms=op_duration,
                    energy_pj=total_energy,
                    chip_id=chip_id,
                    layer_id=layer_idx,
                    microbatch_id=mb_id
                )
                timeline.append(op_exec)
                current_time += op_duration
    
    # 按开始时间排序（虽然已经是顺序的，但保持一致性）
    timeline.sort(key=lambda x: x.start_time_ms)
    return timeline


def compute_windowed_power(
    timeline: List[OperatorExecution],
    window_size_ms: float = 2.0,
    total_duration_ms: float = None
) -> List[TimeWindow]:
    """
    计算时间窗口功耗
    
    Args:
        timeline: 算子执行时间线
        window_size_ms: 时间窗口大小（毫秒）
        total_duration_ms: 总执行时间，如果为None则自动从timeline计算
        
    Returns:
        时间窗口列表，每个包含该窗口的平均功耗
    """
    if not timeline:
        return []
    
    # 确定总执行时间
    if total_duration_ms is None:
        total_duration_ms = max(op.end_time_ms for op in timeline)
    
    # 创建时间窗口
    windows: List[TimeWindow] = []
    current_time = 0.0
    
    while current_time < total_duration_ms:
        window = TimeWindow(current_time, window_size_ms)
        windows.append(window)
        current_time += window_size_ms
    
    # 将每个算子分配到相关的时间窗口
    # 优化：由于算子按时间排序，使用指针避免重复搜索已处理的窗口
    window_start_idx = 0  # 记录上一个算子开始匹配的窗口索引
    
    for op in timeline:
        first_match = True  # 标记是否是第一个匹配的窗口
        
        # 从上一个算子的起始窗口开始查找
        for i in range(window_start_idx, len(windows)):
            window = windows[i]
            
            if op.end_time_ms <= window.start_ms:
                # 算子完全在窗口之前，继续检查下一个窗口
                continue
            
            if op.start_time_ms >= window.end_ms:
                # 算子完全在窗口之后，继续检查下一个窗口
                continue
            
            # 算子与窗口有重叠，添加到窗口
            if first_match:
                window_start_idx = i  # 更新下一个算子的起始搜索位置
                first_match = False
            
            window.add_operator(op)
            # 如果是一个算子的dur_ms小于窗口大小，则不会跨多个窗口m，因此可以break
            if op.end_time_ms <= window.end_ms:
                break
    
    return windows


def analyze_power_statistics(windows: List[TimeWindow]) -> Dict[str, float]:
    """
    分析功耗统计信息
    
    Args:
        windows: 时间窗口列表
        
    Returns:
        统计信息字典
    """
    if not windows:
        return {}
    
    powers = [w.avg_power_w for w in windows]
    
    return {
        '平均功耗(W)': sum(powers) / len(powers) if powers else 0.0,
        '峰值功耗(W)': max(powers) if powers else 0.0,
        '最低功耗(W)': min(powers) if powers else 0.0,
        '功耗标准差(W)': (
            math.sqrt(sum((p - sum(powers)/len(powers))**2 for p in powers) / len(powers))
            if len(powers) > 1 else 0.0
        ),
        '总能耗(J)': sum(w.total_energy_pj for w in windows) * 1e-12,  # pJ to J
        '总时长(ms)': sum(w.duration_ms for w in windows),
    }


def compute_power_from_inference_data(
    prefill_operator_energy: List[Dict],
    decode_operator_energy: List[Dict],
    model_config,
    device_config,
    batch_size: int,
    window_size_ms: float = 2.0,
    leakage_power_w: float = 0.0
) -> Tuple[List[TimeWindow], List[TimeWindow], Dict[str, float]]:
    """
    从inference.py收集的数据计算时间窗口功耗
    
    Args:
        prefill_operator_energy: prefill阶段算子能耗数据（单层单micro-batch）
        decode_operator_energy: decode阶段算子能耗数据（单层单micro-batch）
        model_config: 模型配置
        device_config: 硬件配置
        batch_size: 总batch大小
        window_size_ms: 时间窗口大小（毫秒）
        leakage_power_w: 静态泄漏功耗（瓦特）
        
    Returns:
        (prefill窗口列表, decode窗口列表, 统计信息)
    """
    # 计算pipeline参数
    num_chips = device_config.n_chip
    total_layers = model_config.layer_count
    layers_per_chip = math.ceil(total_layers / num_chips)
    
    # 计算micro-batch数量
    micro_batch_size = batch_size / num_chips
    num_microbatches = num_chips  # 为了充分利用pipeline
    
    # 过滤掉汇总行
    prefill_ops = [op for op in prefill_operator_energy 
                   if op.get('算子') != '汇总' and '总延时(ms)' in op]
    decode_ops = [op for op in decode_operator_energy 
                  if op.get('算子') != '汇总' and '总延时(ms)' in op]
    
    # 构建prefill时间线
    prefill_timeline = build_operator_timeline(
        prefill_ops,
        layers_per_chip=layers_per_chip,
        num_microbatches=num_microbatches,
        num_chips=num_chips,
        leakage_power_w=leakage_power_w
    )
    
    # 构建decode时间线
    decode_timeline = build_operator_timeline(
        decode_ops,
        layers_per_chip=layers_per_chip,
        num_microbatches=num_microbatches,
        num_chips=num_chips,
        leakage_power_w=leakage_power_w
    )
    
    # 计算时间窗口功耗
    prefill_windows = compute_windowed_power(prefill_timeline, window_size_ms)
    decode_windows = compute_windowed_power(decode_timeline, window_size_ms)
    
    # 分析统计信息
    prefill_stats = analyze_power_statistics(prefill_windows)
    decode_stats = analyze_power_statistics(decode_windows)
    
    # 合并统计
    combined_stats = {
        'Prefill平均功耗(W)': prefill_stats.get('平均功耗(W)', 0.0),
        'Prefill峰值功耗(W)': prefill_stats.get('峰值功耗(W)', 0.0),
        'Prefill总能耗(J)': prefill_stats.get('总能耗(J)', 0.0),
        'Prefill总时长(ms)': prefill_stats.get('总时长(ms)', 0.0),
        'Decode平均功耗(W)': decode_stats.get('平均功耗(W)', 0.0),
        'Decode峰值功耗(W)': decode_stats.get('峰值功耗(W)', 0.0),
        'Decode总能耗(J)': decode_stats.get('总能耗(J)', 0.0),
        'Decode总时长(ms)': decode_stats.get('总时长(ms)', 0.0),
        '时间窗口大小(ms)': window_size_ms,
        'Prefill窗口数': len(prefill_windows),
        'Decode窗口数': len(decode_windows),
    }
    
    return prefill_windows, decode_windows, combined_stats
