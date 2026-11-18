# ===============================================================================
# stockpyl - Unit Tests for Trace-Driven Demand Replay
# ===============================================================================

import unittest
from stockpyl.demand_source import DemandSource
from stockpyl.supply_chain_network import single_stage_system
from stockpyl.sim import simulation


class TestTraceDemandSource(unittest.TestCase):
	"""Test TRACE demand source behavior and simulation replay."""

	def test_trace_properties(self):
		history = [12.0, 18.0, 15.0, 25.0, 30.0]
		ds = DemandSource(type='TRACE', demand_list=history)
		self.assertEqual(ds.type, 'TRACE')
		self.assertAlmostEqual(ds.mean, 20.0)
		self.assertGreater(ds.standard_deviation, 0.0)

	def test_trace_generation_cycles(self):
		history = [10.0, 20.0, 30.0]
		ds = DemandSource(type='TRACE', demand_list=history)
		# Sequential periods
		self.assertEqual(ds.generate_demand(period=0), 10.0)
		self.assertEqual(ds.generate_demand(period=1), 20.0)
		self.assertEqual(ds.generate_demand(period=2), 30.0)
		# Beyond length cycles
		self.assertEqual(ds.generate_demand(period=3), 10.0)
		self.assertEqual(ds.generate_demand(period=4), 20.0)

	def test_trace_simulation_integration(self):
		history = [5.0, 10.0, 15.0, 20.0, 25.0]
		ds = DemandSource(type='TRACE', demand_list=history)
		network = single_stage_system(
			holding_cost=1.0,
			stockout_cost=10.0,
			policy_type='BS',
			base_stock_level=30.0,
			lead_time=1,
			demand_source=ds
		)
		# Run 5-period simulation
		total_cost = simulation(network, num_periods=5, progress_bar=False)
		self.assertGreater(total_cost, 0.0)
		node = network.nodes[0]
		# Demands generated should match the trace
		observed_demands = [node.state_vars[t].inbound_order[None][node.product_indices[0]] for t in range(5)]
		self.assertEqual(observed_demands, history)


if __name__ == '__main__':
	unittest.main()
