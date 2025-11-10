from utils import DataType
from utils import Tensor
from utils import data_type_dict
from utils import size
from hardware_model.device import Device
from power import get_global_power_counter
import math

def reduce_multicast(device:Device, op_name: str = None):    #小数据量的reduce-multicast
    noc_link_latency = device.communication.onchip_communacation.noc_link_latency
    noc_router_latency = device.communication.onchip_communacation.noc_router_latency

    return 2 * (noc_link_latency + (device.row_core_count+device.col_core_count-2) * noc_router_latency)

#每个核都全部广播，实现一次All-Gather
def multicast(device:Device ,traffic: int, op_name: str = None):
    noc_link_latency = device.communication.onchip_communacation.noc_link_latency
    noc_router_latency = device.communication.onchip_communacation.noc_router_latency
    noc_bandwidth_per_cycle = device.communication.onchip_communacation.noc_bandwidth_per_cycle

    latency = (device.row_core_count+device.col_core_count-2)*(traffic/noc_bandwidth_per_cycle + noc_router_latency) + noc_link_latency
    # power accounting: approximate avg hops per packet equals (row+col-2)
    hops = (device.row_core_count + device.col_core_count - 2)
    counter = get_global_power_counter()
    counter.add_noc_transfer(bytes_transferred=traffic, avg_hops=hops, operator=op_name or "multicast")
    return latency

def concat(device:Device ,traffic: int, op_name: str = None):
    noc_link_latency = device.communication.onchip_communacation.noc_link_latency
    noc_router_latency = device.communication.onchip_communacation.noc_router_latency
    noc_bandwidth_per_cycle = device.communication.onchip_communacation.noc_bandwidth_per_cycle

    latency = noc_link_latency + (device.row_core_count-1)*(traffic/ device.core_count/noc_bandwidth_per_cycle + noc_router_latency)\
                + (device.row_core_count-1)*(device.row_core_count*(traffic/ device.core_count/noc_bandwidth_per_cycle) + noc_router_latency)
    # power accounting: estimate bit-hops as traffic * (row_core_count-1)
    hops = max(device.row_core_count - 1, 1)
    counter = get_global_power_counter()
    counter.add_noc_transfer(bytes_transferred=traffic, avg_hops=hops, operator=op_name or "concat")
    return latency

def scatter(device:Device ,traffic: int, op_name: str = None):
    noc_link_latency = device.communication.onchip_communacation.noc_link_latency
    noc_router_latency = device.communication.onchip_communacation.noc_router_latency
    noc_bandwidth_per_cycle = device.communication.onchip_communacation.noc_bandwidth_per_cycle

    latency = noc_link_latency + (device.row_core_count-1)*(traffic/ device.core_count/noc_bandwidth_per_cycle + noc_router_latency)\
                + (device.row_core_count-1)*(device.row_core_count*(traffic/ device.core_count/noc_bandwidth_per_cycle) + noc_router_latency)
    hops = max(device.row_core_count - 1, 1)
    counter = get_global_power_counter()
    counter.add_noc_transfer(bytes_transferred=traffic, avg_hops=hops, operator=op_name or "scatter")
    return latency

def p2p(device:Device ,traffic: int, op_name: str = None):
    scale_bandwidth = device.communication.scale_out.scale_out_bandwidth_per_cycle
    scale_link_latency = device.communication.scale_out.link_latency
    latency = scale_link_latency + traffic / scale_bandwidth
    # power accounting: treat as PCIe/scale-out per-bit energy
    counter = get_global_power_counter()
    counter.add_pcie_bytes(byte_count=traffic, operator=op_name or "p2p")
    return latency