"""
测试时间窗口功耗计算功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_inference.compute_power import (
    OperatorExecution,
    TimeWindow,
    build_operator_timeline,
    compute_windowed_power,
    analyze_power_statistics
)

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试1: 基本数据结构")
    print("=" * 60)
    
    # 创建算子执行记录（使用更真实的能耗值）
    op1 = OperatorExecution(
        name="GEMM_Q",
        start_time_ms=0.0,
        duration_ms=1.5,
        energy_pj=3000000000.0,  # 3e9 pJ = 3 J
        chip_id=0,
        layer_id=0,
        microbatch_id=0
    )
    
    print(f"算子: {op1.name}")
    print(f"  开始时间: {op1.start_time_ms} ms")
    print(f"  持续时间: {op1.duration_ms} ms")
    print(f"  结束时间: {op1.end_time_ms} ms")
    print(f"  能耗: {op1.energy_pj} pJ")
    print(f"  平均功耗: {op1.avg_power_w:.3f} W")
    print()
    
    # 创建时间窗口
    window = TimeWindow(start_ms=0.0, duration_ms=2.0)
    window.add_operator(op1)
    
    print(f"时间窗口: {window.start_ms}-{window.end_ms} ms")
    print(f"  窗口能耗: {window.total_energy_pj} pJ")
    print(f"  平均功耗: {window.avg_power_w:.3f} W")
    print()

def test_operator_timeline():
    """测试时间线构建"""
    print("=" * 60)
    print("测试2: 算子时间线构建")
    print("=" * 60)
    
    # 模拟单层单micro-batch的算子数据（更真实的能耗值）
    operator_data = [
        {'operator': 'GEMM_Q', '总延时(ms)': 0.5, '能耗(pJ)': 1000000000.0},  # 1J
        {'operator': 'GEMM_K', '总延时(ms)': 0.3, '能耗(pJ)': 600000000.0},   # 0.6J
        {'operator': 'Softmax', '总延时(ms)': 0.2, '能耗(pJ)': 400000000.0},  # 0.4J
    ]
    
    timeline = build_operator_timeline(
        operator_data=operator_data,
        layers_per_chip=1,
        num_microbatches=2,
        num_chips=2,
        leakage_power_w=0.0
    )
    
    print(f"生成了 {len(timeline)} 个算子执行记录")
    print("\n前10个算子执行记录:")
    for i, op in enumerate(timeline[:10]):
        print(f"  {i+1}. {op.name:12s} Chip{op.chip_id} MB{op.microbatch_id} "
              f"[{op.start_time_ms:.2f}-{op.end_time_ms:.2f}ms] "
              f"{op.energy_pj:.1f}pJ {op.avg_power_w:.3f}W")
    print()

def test_windowed_power():
    """测试时间窗口功耗计算"""
    print("=" * 60)
    print("测试3: 时间窗口功耗计算")
    print("=" * 60)
    
    # 创建简单的时间线（更真实的能耗值）
    operator_data = [
        {'operator': 'GEMM', '总延时(ms)': 1.5, '能耗(pJ)': 3000000000.0},  # 3J
        {'operator': 'Add', '总延时(ms)': 0.5, '能耗(pJ)': 500000000.0},    # 0.5J
    ]
    
    timeline = build_operator_timeline(
        operator_data=operator_data,
        layers_per_chip=1,
        num_microbatches=2,
        num_chips=2,
        leakage_power_w=10.0  # 10W泄漏功耗
    )
    
    windows = compute_windowed_power(
        timeline=timeline,
        window_size_ms=2.0
    )
    
    print(f"生成了 {len(windows)} 个时间窗口")
    print("\n时间窗口功耗:")
    for i, w in enumerate(windows):
        print(f"  窗口{i}: [{w.start_ms:.2f}-{w.end_ms:.2f}ms] "
              f"能耗={w.total_energy_pj*1e-12:.6f}J "
              f"功耗={w.avg_power_w:.3f}W "
              f"算子数={len(w.operators)}")
    print()

def test_power_statistics():
    """测试功耗统计"""
    print("=" * 60)
    print("测试4: 功耗统计分析")
    print("=" * 60)
    
    operator_data = [
        {'operator': 'GEMM', '总延时(ms)': 1.0, '能耗(pJ)': 2000000000.0},  # 2J
        {'operator': 'Add', '总延时(ms)': 0.5, '能耗(pJ)': 500000000.0},    # 0.5J
    ]
    
    timeline = build_operator_timeline(
        operator_data=operator_data,
        layers_per_chip=2,
        num_microbatches=4,
        num_chips=4,
        leakage_power_w=5.0
    )
    
    windows = compute_windowed_power(timeline, window_size_ms=1.0)
    stats = analyze_power_statistics(windows)
    
    print("功耗统计:")
    for key, value in stats.items():
        if 'W' in key or '功耗' in key:
            print(f"  {key:20s}: {value:.3f}")
        else:
            print(f"  {key:20s}: {value:.6f}")
    print()

def main():
    print("\n" + "=" * 60)
    print("时间窗口功耗计算模块测试")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_operator_timeline()
        test_windowed_power()
        test_power_statistics()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
