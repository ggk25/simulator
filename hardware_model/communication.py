class onchip_communacation:
	def __init__(
		self,
		noc_link_latency,	#cycles
		noc_router_latency,	#cycles
		noc_bandwidth,	#GB/s
		frequency,
	):
		self.noc_link_latency = noc_link_latency
		self.noc_router_latency = noc_router_latency
		self.noc_bandwidth = noc_bandwidth
		self.noc_bandwidth_per_cycle = noc_bandwidth / frequency * 1e3

class scale_up:
	def __init__(
		self,
		link_latency,	#cycles
		scale_up_bandwidth,	#GB/s
		frequency

	):
		self.link_latency = link_latency
		self.scale_up_bandwidth = scale_up_bandwidth
		self.scale_up_bandwidth_per_cycle = scale_up_bandwidth / frequency * 1e3


class scale_out:
	def __init__(
		self,
		link_latency,	#cycles
		scale_out_bandwidth,	#GB/s
		frequency
	):
		self.link_latency = link_latency
		self.scale_out_bandwidth = scale_out_bandwidth
		self.scale_out_bandwidth_per_cycle = scale_out_bandwidth / frequency * 1e3


class communication:
	def __init__(
		self,
		onchip_communacation:onchip_communacation,
		scale_up:scale_up,
		scale_out:scale_out,
	)-> None:
		self.onchip_communacation = onchip_communacation
		self.scale_up = scale_up
		self.scale_out = scale_out

communication_dict = {
	"D37x": communication(
		onchip_communacation(30, 3, 128, 800),
		scale_up(50, 32, 800),
		scale_out(100, 16, 800),
	),
}
