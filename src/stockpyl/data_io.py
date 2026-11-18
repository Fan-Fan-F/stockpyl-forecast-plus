# ===============================================================================
# stockpyl - E-commerce Order Data I/O and Aggregation Engine
# -------------------------------------------------------------------------------
# Author: Larry Snyder, stockpyl contributors
# License: MIT
# ===============================================================================

"""
Overview
--------

The ``data_io`` module provides robust import, normalization, and aggregation
interfaces for e-commerce order streams (CSV, JSON, pandas DataFrame, list of dicts),
converting raw transactional logs into continuous time series and native
``DemandSource`` objects for inventory simulation and optimization.

Key Features:
  * Transaction-level order ingestion with customizable schema mapping
  * Zero-demand padding: automatically generates zeros on days without sales
  * Grouped resampling by SKU, category, and fulfillment center (daily, weekly, monthly)
  * Seamless conversion of order history into stockpyl ``DemandSource`` (Normal, Poisson, Trace)
  * Rapid creation of multi-echelon e-commerce fulfillment networks (CDC -> RDCs/FCs)
"""

import os
import csv
import datetime
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np

from stockpyl.demand_source import DemandSource
from stockpyl.supply_chain_network import network_from_edges, SupplyChainNetwork
from stockpyl.policy import Policy
from stockpyl.forecast import auto_forecast, forecast_to_inventory_params


# ===============================================================================
# Order Ingestion and Normalization
# ===============================================================================

def parse_date(val: Any) -> datetime.date:
	"""Parse an arbitrary date representation into datetime.date."""
	if isinstance(val, datetime.datetime):
		return val.date()
	if isinstance(val, datetime.date):
		return val
	val_str = str(val).strip()
	for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%Y%m%d'):
		try:
			return datetime.datetime.strptime(val_str, fmt).date()
		except ValueError:
			pass
	raise ValueError(f"Could not parse date from '{val}'")


def load_order_history(
	source: Union[str, List[Dict[str, Any]], Any],
	date_col: str = 'order_date',
	sku_col: str = 'sku_id',
	qty_col: str = 'quantity',
	node_col: Optional[str] = 'fulfillment_node',
	price_col: Optional[str] = 'unit_price'
) -> List[Dict[str, Any]]:
	"""Load and normalize transactional order history from CSV, DataFrame, or dict list.

	Parameters
	----------
	source : str, DataFrame, or list of dict
		Path to CSV file, or in-memory collection of order records.
	date_col : str
		Name of the column containing order timestamps/dates.
	sku_col : str
		Name of the SKU / product identifier column.
	qty_col : str
		Name of the ordered quantity column.
	node_col : str, optional
		Name of fulfillment warehouse or node column.
	price_col : str, optional
		Name of item price column.

	Returns
	-------
	List[Dict[str, Any]]
		Standardized order records with parsed date, float quantity, and string SKU.
	"""
	raw_records = []
	if isinstance(source, str):
		if not os.path.exists(source):
			raise FileNotFoundError(f"Order data file not found at '{source}'")
		with open(source, mode='r', encoding='utf-8', errors='replace') as f:
			reader = csv.DictReader(f)
			raw_records = list(reader)
	elif hasattr(source, 'to_dict'):
		# pandas DataFrame
		raw_records = source.to_dict(orient='records')
	elif isinstance(source, list):
		raw_records = source
	else:
		raise TypeError(f"Unsupported source type: {type(source)}")

	cleaned = []
	for idx, row in enumerate(raw_records):
		if date_col not in row or sku_col not in row or qty_col not in row:
			raise KeyError(f"Row {idx} missing required columns: {date_col}, {sku_col}, or {qty_col}")

		try:
			order_date = parse_date(row[date_col])
			quantity = float(row[qty_col])
		except (ValueError, TypeError) as e:
			continue  # Skip corrupt row

		record = {
			'order_date': order_date,
			'sku_id': str(row[sku_col]).strip(),
			'quantity': quantity
		}
		if node_col and node_col in row and row[node_col] is not None:
			record['fulfillment_node'] = str(row[node_col]).strip()
		if price_col and price_col in row and row[price_col] is not None:
			try:
				record['unit_price'] = float(row[price_col])
			except (ValueError, TypeError):
				pass
		cleaned.append(record)

	return cleaned


def aggregate_orders(
	orders: List[Dict[str, Any]],
	freq: str = 'D',
	start_date: Optional[Union[str, datetime.date]] = None,
	end_date: Optional[Union[str, datetime.date]] = None,
	group_by_node: bool = False
) -> Dict[Any, np.ndarray]:
	"""Aggregate discrete orders into continuous periodic demand arrays with zero-filling.

	Parameters
	----------
	orders : List[Dict[str, Any]]
		Normalized order records from ``load_order_history()``.
	freq : str, optional
		Aggregation frequency: 'D' (daily, default), 'W' (weekly).
	start_date : date or str, optional
		Start of time window. If None, derived from min(order_date).
	end_date : date or str, optional
		End of time window. If None, derived from max(order_date).
	group_by_node : bool, optional
		If True, keys of result are (fulfillment_node, sku_id). Otherwise sku_id.

	Returns
	-------
	Dict[Any, np.ndarray]
		Mapping from SKU (or (node, SKU)) to continuous periodic demand array.
	"""
	if not orders:
		return {}

	parsed_start = parse_date(start_date) if start_date else min(r['order_date'] for r in orders)
	parsed_end = parse_date(end_date) if end_date else max(r['order_date'] for r in orders)

	if parsed_start > parsed_end:
		raise ValueError("start_date cannot be later than end_date")

	step_days = 7 if freq.upper() == 'W' else 1
	num_periods = ((parsed_end - parsed_start).days // step_days) + 1

	grouped_quantities: Dict[Any, np.ndarray] = {}

	for record in orders:
		r_date = record['order_date']
		if r_date < parsed_start or r_date > parsed_end:
			continue

		period_idx = (r_date - parsed_start).days // step_days
		if period_idx >= num_periods:
			period_idx = num_periods - 1

		sku = record['sku_id']
		if group_by_node and 'fulfillment_node' in record:
			key = (record['fulfillment_node'], sku)
		else:
			key = sku

		if key not in grouped_quantities:
			grouped_quantities[key] = np.zeros(num_periods, dtype=float)

		grouped_quantities[key][period_idx] += record['quantity']

	return grouped_quantities


# ===============================================================================
# Conversion to stockpyl DemandSource and Networks
# ===============================================================================

def orders_to_demand_source(
	demand_series: Union[List[float], np.ndarray],
	demand_type: str = 'N',
	round_to_int: bool = True
) -> DemandSource:
	"""Convert an empirical order demand series into a stockpyl ``DemandSource``.

	Parameters
	----------
	demand_series : List[float] or np.ndarray
		Periodic order quantities.
	demand_type : str
		Target distribution type:
		  * 'N': Normal distribution fitted via empirical sample mean and std
		  * 'P': Poisson distribution fitted via empirical sample mean
		  * 'TRACE': Trace-driven demand replay of historical sequence
		  * 'D': Deterministic cyclic sequence
	round_to_int : bool
		Whether generated demands should be rounded to integers.

	Returns
	-------
	DemandSource
		Configured stockpyl DemandSource object.
	"""
	arr = np.asarray(demand_series, dtype=float)
	if len(arr) == 0:
		raise ValueError("demand_series is empty")

	dtype = demand_type.upper()
	if dtype == 'N':
		m = float(np.mean(arr))
		s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
		# Avoid 0 std causing errors in normal distribution
		s = max(1e-3, s)
		return DemandSource(type='N', mean=m, standard_deviation=s, round_to_int=round_to_int)

	elif dtype == 'P':
		m = max(1e-3, float(np.mean(arr)))
		return DemandSource(type='P', mean=m, round_to_int=round_to_int)

	elif dtype == 'TRACE':
		return DemandSource(type='TRACE', demand_list=arr.tolist(), round_to_int=round_to_int)

	elif dtype == 'D':
		return DemandSource(type='D', demand_list=arr.tolist(), round_to_int=round_to_int)

	else:
		raise ValueError(f"Unsupported demand_type '{demand_type}' for order conversion. Choose from 'N', 'P', 'TRACE', 'D'.")


def load_sample_ecommerce_orders() -> List[Dict[str, Any]]:
	"""Load the bundled real-world e-commerce order dataset.

	Returns
	-------
	List[Dict[str, Any]]
		Parsed order records covering 3 distinct SKU demand patterns across 2 fulfillment centers.
	"""
	pkg_dir = os.path.dirname(os.path.abspath(__file__))
	csv_path = os.path.join(pkg_dir, 'datasets', 'ecommerce_orders.csv')
	if not os.path.exists(csv_path):
		# Fall back to workspace datasets directory
		alt_path = os.path.join(os.path.dirname(pkg_dir), '..', 'datasets', 'ecommerce_orders.csv')
		if os.path.exists(alt_path):
			csv_path = alt_path
	return load_order_history(csv_path)


def build_ecommerce_fulfillment_network(
	sku_id: str,
	daily_demand_series: Union[List[float], np.ndarray],
	service_level: float = 0.95,
	supplier_lead_time: int = 3,
	fulfillment_lead_time: int = 1,
	holding_cost_cdc: float = 1.0,
	holding_cost_fc: float = 2.0,
	stockout_cost_fc: float = 20.0,
	fixed_cost_order: float = 30.0,
	policy_type: str = 'BS',
	demand_mode: str = 'N'
) -> SupplyChainNetwork:
	"""Quickly construct a 2-echelon e-commerce fulfillment network:
	External Supplier -> Central Distribution Center (CDC, Node 1) -> Regional Fulfillment Center (FC, Node 2) -> Customer Demand.

	Parameters
	----------
	sku_id : str
		SKU identifier for logging and naming.
	daily_demand_series : list or array
		Historical or forecast daily demand time series at the fulfillment center.
	service_level : float
		Target cycle service level (e.g. 0.95).
	supplier_lead_time : int
		Lead time from supplier to CDC in days.
	fulfillment_lead_time : int
		Lead time from CDC to FC in days.
	holding_cost_cdc : float
		Daily holding cost per unit at CDC.
	holding_cost_fc : float
		Daily holding cost per unit at FC.
	stockout_cost_fc : float
		Stockout penalty cost per unit at FC.
	fixed_cost_order : float
		Fixed order setup cost.
	policy_type : str
		'BS' (Base Stock) or 'rQ'.
	demand_mode : str
		'N' (Normal fitted) or 'TRACE' (Empirical trace replay).

	Returns
	-------
	SupplyChainNetwork
		Fully configured stockpyl SupplyChainNetwork instance ready for simulation.
	"""
	data = np.asarray(daily_demand_series, dtype=float)
	ds = orders_to_demand_source(data, demand_type=demand_mode, round_to_int=True)

	# Compute FC inventory policy parameters based on demand forecast
	fc_params = forecast_to_inventory_params(
		mean_demand=ds.mean or float(np.mean(data)),
		residual_std=ds.standard_deviation or float(np.std(data)),
		lead_time=float(fulfillment_lead_time),
		service_level=service_level,
		holding_cost=holding_cost_fc,
		fixed_cost=fixed_cost_order
	)
	fc_policy = fc_params.to_policy(policy_type)

	# Compute CDC inventory policy parameters (covers echelon lead time)
	cdc_params = forecast_to_inventory_params(
		mean_demand=fc_params.mean_demand_per_period,
		residual_std=fc_params.lead_time_demand_std,
		lead_time=float(supplier_lead_time),
		service_level=service_level,
		holding_cost=holding_cost_cdc,
		fixed_cost=fixed_cost_order * 1.5
	)
	cdc_policy = cdc_params.to_policy(policy_type)

	network = network_from_edges(
		edges=[(1, 2)],  # Node 1 (CDC) supplies Node 2 (FC)
		node_order_in_lists=[1, 2],
		local_holding_cost=[holding_cost_cdc, holding_cost_fc],
		stockout_cost=[0.0, stockout_cost_fc],
		shipment_lead_time=[supplier_lead_time, fulfillment_lead_time],
		demand_source=[None, ds],
		inventory_policy=[cdc_policy, fc_policy]
	)

	return network
