# ===============================================================================
# stockpyl - Inventory Optimization, Forecasting, and Simulation
# -------------------------------------------------------------------------------
# Author: Larry Snyder, stockpyl contributors
# License: MIT
# ===============================================================================

"""
stockpyl
========
A Python library for inventory optimization, multi-echelon supply chain simulation,
and e-commerce demand forecasting.
"""

__version__ = "1.1.0"

# Core inventory & simulation modules
from stockpyl.supply_chain_network import SupplyChainNetwork, network_from_edges, single_stage_system, serial_system
from stockpyl.supply_chain_node import SupplyChainNode
from stockpyl.supply_chain_product import SupplyChainProduct
from stockpyl.demand_source import DemandSource
from stockpyl.policy import Policy
from stockpyl.sim import simulation

# E-commerce forecasting and data ingestion modules
from stockpyl import forecast
from stockpyl import data_io
