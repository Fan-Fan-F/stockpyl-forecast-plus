# ===============================================================================
# stockpyl - Unit Tests for E-commerce Data I/O and Aggregation
# ===============================================================================

import unittest
import datetime
import numpy as np

from stockpyl.data_io import (
	parse_date, load_order_history, aggregate_orders,
	orders_to_demand_source, load_sample_ecommerce_orders,
	build_ecommerce_fulfillment_network
)
from stockpyl.demand_source import DemandSource
from stockpyl.sim import simulation


class TestDataIO(unittest.TestCase):
	"""Test data ingestion and normalization functions."""

	def test_parse_date(self):
		self.assertEqual(parse_date("2026-01-15"), datetime.date(2026, 1, 15))
		self.assertEqual(parse_date("2026/02/20"), datetime.date(2026, 2, 20))
		self.assertEqual(parse_date("03/15/2026"), datetime.date(2026, 3, 15))
		dt = datetime.datetime(2026, 5, 1, 14, 30, 0)
		self.assertEqual(parse_date(dt), datetime.date(2026, 5, 1))

	def test_load_order_history_from_list(self):
		records = [
			{'order_date': '2026-01-01', 'sku_id': 'SKU-A', 'quantity': 10, 'fulfillment_node': 'FC1'},
			{'order_date': '2026-01-02', 'sku_id': 'SKU-A', 'quantity': 15, 'fulfillment_node': 'FC1'},
			{'order_date': 'invalid-date', 'sku_id': 'SKU-A', 'quantity': 5},  # should be skipped
			{'order_date': '2026-01-03', 'sku_id': 'SKU-B', 'quantity': 25, 'fulfillment_node': 'FC2'}
		]
		cleaned = load_order_history(records)
		self.assertEqual(len(cleaned), 3)
		self.assertEqual(cleaned[0]['sku_id'], 'SKU-A')
		self.assertEqual(cleaned[0]['quantity'], 10.0)
		self.assertEqual(cleaned[2]['sku_id'], 'SKU-B')

	def test_load_sample_ecommerce_orders(self):
		orders = load_sample_ecommerce_orders()
		self.assertGreater(len(orders), 500)
		skus = {o['sku_id'] for o in orders}
		self.assertIn('SKU-FAST-101', skus)
		self.assertIn('SKU-SEAS-202', skus)
		self.assertIn('SKU-SLOW-303', skus)

	def test_aggregate_orders_daily_and_zero_fill(self):
		orders = [
			{'order_date': datetime.date(2026, 1, 1), 'sku_id': 'SKU-1', 'quantity': 10},
			{'order_date': datetime.date(2026, 1, 1), 'sku_id': 'SKU-1', 'quantity': 5},
			# 2026-01-02 has NO orders (must be filled with 0)
			{'order_date': datetime.date(2026, 1, 3), 'sku_id': 'SKU-1', 'quantity': 20}
		]
		agg = aggregate_orders(orders, freq='D', start_date='2026-01-01', end_date='2026-01-03')
		self.assertIn('SKU-1', agg)
		arr = agg['SKU-1']
		self.assertEqual(len(arr), 3)
		self.assertEqual(arr[0], 15.0)  # Day 1: 10 + 5
		self.assertEqual(arr[1], 0.0)   # Day 2: 0 (zero-filled!)
		self.assertEqual(arr[2], 20.0)  # Day 3: 20

	def test_aggregate_orders_by_node(self):
		orders = [
			{'order_date': datetime.date(2026, 1, 1), 'sku_id': 'SKU-1', 'quantity': 10, 'fulfillment_node': 'FC_EAST'},
			{'order_date': datetime.date(2026, 1, 1), 'sku_id': 'SKU-1', 'quantity': 20, 'fulfillment_node': 'FC_WEST'}
		]
		agg = aggregate_orders(orders, group_by_node=True)
		self.assertIn(('FC_EAST', 'SKU-1'), agg)
		self.assertIn(('FC_WEST', 'SKU-1'), agg)
		self.assertEqual(agg[('FC_EAST', 'SKU-1')][0], 10.0)
		self.assertEqual(agg[('FC_WEST', 'SKU-1')][0], 20.0)

	def test_orders_to_demand_source(self):
		demand_series = [10, 15, 20, 15, 25, 30]
		# Normal distribution
		ds_n = orders_to_demand_source(demand_series, demand_type='N')
		self.assertEqual(ds_n.type, 'N')
		self.assertAlmostEqual(ds_n.mean, np.mean(demand_series))

		# Poisson distribution
		ds_p = orders_to_demand_source(demand_series, demand_type='P')
		self.assertEqual(ds_p.type, 'P')
		self.assertAlmostEqual(ds_p.mean, np.mean(demand_series))

		# Trace distribution
		ds_trace = orders_to_demand_source(demand_series, demand_type='TRACE')
		self.assertEqual(ds_trace.type, 'TRACE')
		self.assertEqual(ds_trace.generate_demand(period=0), 10.0)
		self.assertEqual(ds_trace.generate_demand(period=2), 20.0)

	def test_build_ecommerce_fulfillment_network(self):
		daily_demand = [40, 55, 60, 45, 50, 65, 70, 48, 52, 58]
		network = build_ecommerce_fulfillment_network(
			sku_id='SKU-FAST-101',
			daily_demand_series=daily_demand,
			service_level=0.95,
			supplier_lead_time=2,
			fulfillment_lead_time=1,
			demand_mode='N'
		)

		self.assertEqual(len(network.nodes), 2)
		cdc_node = network.nodes_by_index[1]
		fc_node = network.nodes_by_index[2]

		# Verify network structure: CDC has no demand, FC has demand source
		self.assertIsNone(cdc_node.demand_source.type)
		self.assertEqual(fc_node.demand_source.type, 'N')

		# Verify simulation runs smoothly through the network
		total_cost = simulation(network, num_periods=10, progress_bar=False)
		self.assertGreater(total_cost, 0.0)


if __name__ == '__main__':
	unittest.main()
