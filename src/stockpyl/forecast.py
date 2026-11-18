# ===============================================================================
# stockpyl - Demand Forecasting Module for E-commerce Order Streams
# -------------------------------------------------------------------------------
# Author: Larry Snyder, stockpyl contributors
# License: MIT
# ===============================================================================

"""
Overview
--------

The ``forecast`` module provides time series demand forecasting methods tailored
for e-commerce order histories, ranging from fast-moving retail SKUs with seasonal
trends to slow-moving intermittent spare parts.

Supported forecasting algorithms include:
  * Simple Moving Average (SMA) and Weighted Moving Average (WMA)
  * Simple Exponential Smoothing (SES) with automated alpha parameter optimization
  * Holt's Linear Trend Model (double exponential smoothing)
  * Holt-Winters Additive and Multiplicative Seasonality Models
  * Croston's Method for intermittent e-commerce demand
  * Syntetos-Boylan Approximation (SBA) for unbiased intermittent demand estimation

In addition, it provides accuracy evaluation metrics (MAE, RMSE, MAPE, WAPE, SMAPE)
and a direct bridge connecting forecast mean and residual variance to inventory
control parameters (Safety Stock, Reorder Point, Base Stock Level, EOQ).
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict
import math
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from stockpyl.policy import Policy


# ===============================================================================
# Error Metrics
# ===============================================================================

def mae(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
	"""Calculate Mean Absolute Error (MAE)."""
	y_t = np.asarray(y_true, dtype=float)
	y_p = np.asarray(y_pred, dtype=float)
	if len(y_t) == 0:
		return 0.0
	return float(np.mean(np.abs(y_t - y_p)))


def rmse(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
	"""Calculate Root Mean Squared Error (RMSE)."""
	y_t = np.asarray(y_true, dtype=float)
	y_p = np.asarray(y_pred, dtype=float)
	if len(y_t) == 0:
		return 0.0
	return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def wape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
	"""Calculate Weighted Absolute Percentage Error (WAPE).

	WAPE = sum(|y - y_hat|) / sum(y)
	Robust against zero-demand periods common in retail and e-commerce.
	"""
	y_t = np.asarray(y_true, dtype=float)
	y_p = np.asarray(y_pred, dtype=float)
	total_actual = np.sum(np.abs(y_t))
	if total_actual == 0:
		return 0.0 if np.sum(np.abs(y_p)) == 0 else 1.0
	return float(np.sum(np.abs(y_t - y_p)) / total_actual)


def mape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], eps: float = 1e-5) -> float:
	"""Calculate Mean Absolute Percentage Error (MAPE)."""
	y_t = np.asarray(y_true, dtype=float)
	y_p = np.asarray(y_pred, dtype=float)
	mask = np.abs(y_t) > eps
	if not np.any(mask):
		return 0.0
	return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])))


def smape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
	"""Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
	y_t = np.asarray(y_true, dtype=float)
	y_p = np.asarray(y_pred, dtype=float)
	denom = np.abs(y_t) + np.abs(y_p)
	mask = denom > 1e-8
	if not np.any(mask):
		return 0.0
	return float(np.mean(2.0 * np.abs(y_t[mask] - y_p[mask]) / denom[mask]))


def evaluate_forecast(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> Dict[str, float]:
	"""Evaluate forecasting accuracy across multiple standard metrics."""
	return {
		'mae': mae(y_true, y_pred),
		'rmse': rmse(y_true, y_pred),
		'wape': wape(y_true, y_pred),
		'mape': mape(y_true, y_pred),
		'smape': smape(y_true, y_pred)
	}


# ===============================================================================
# Forecasting Models
# ===============================================================================

@dataclass
class ForecastResult:
	"""Structured result of a time-series forecast."""
	method: str
	forecasts: np.ndarray
	fitted_values: np.ndarray
	residuals: np.ndarray
	parameters: dict
	metrics: dict

	@property
	def mean_forecast(self) -> float:
		"""Mean of the forecasted periods."""
		return float(np.mean(self.forecasts)) if len(self.forecasts) > 0 else 0.0

	@property
	def residual_std(self) -> float:
		"""Sample standard deviation of historical in-sample residuals."""
		valid_residuals = self.residuals[~np.isnan(self.residuals)]
		if len(valid_residuals) < 2:
			return float(np.std(valid_residuals)) if len(valid_residuals) == 1 else 0.0
		return float(np.std(valid_residuals, ddof=1))


def simple_moving_average(
	history: Union[List[float], np.ndarray],
	window: int = 3,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Compute Simple Moving Average (SMA) forecast."""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n < 1:
		raise ValueError("history must contain at least 1 observation")
	if window < 1:
		raise ValueError("window must be at least 1")
	actual_window = min(window, n)

	fitted = np.full(n, np.nan, dtype=float)
	for t in range(actual_window, n):
		fitted[t] = np.mean(data[t - actual_window:t])

	last_val = np.mean(data[-actual_window:])
	forecasts = np.full(forecast_periods, last_val, dtype=float)
	valid_idx = ~np.isnan(fitted)
	residuals = np.full(n, np.nan, dtype=float)
	if np.any(valid_idx):
		residuals[valid_idx] = data[valid_idx] - fitted[valid_idx]
		metrics = evaluate_forecast(data[valid_idx], fitted[valid_idx])
	else:
		metrics = {'mae': 0.0, 'rmse': 0.0, 'wape': 0.0, 'mape': 0.0, 'smape': 0.0}

	return ForecastResult(
		method='SMA',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={'window': window},
		metrics=metrics
	)


def weighted_moving_average(
	history: Union[List[float], np.ndarray],
	weights: Optional[Union[List[float], np.ndarray]] = None,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Compute Weighted Moving Average (WMA) forecast."""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n < 1:
		raise ValueError("history must contain at least 1 observation")

	if weights is None:
		window = min(3, n)
		weights = np.arange(1, window + 1, dtype=float)
	else:
		weights = np.asarray(weights, dtype=float)
		window = len(weights)
	
	if window > n:
		weights = weights[-n:]
		window = n
	w = weights / np.sum(weights)

	fitted = np.full(n, np.nan, dtype=float)
	for t in range(window, n):
		fitted[t] = np.dot(w, data[t - window:t])

	last_val = float(np.dot(w, data[-window:]))
	forecasts = np.full(forecast_periods, last_val, dtype=float)

	valid_idx = ~np.isnan(fitted)
	residuals = np.full(n, np.nan, dtype=float)
	if np.any(valid_idx):
		residuals[valid_idx] = data[valid_idx] - fitted[valid_idx]
		metrics = evaluate_forecast(data[valid_idx], fitted[valid_idx])
	else:
		metrics = {'mae': 0.0, 'rmse': 0.0, 'wape': 0.0, 'mape': 0.0, 'smape': 0.0}

	return ForecastResult(
		method='WMA',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={'weights': weights.tolist()},
		metrics=metrics
	)


def simple_exponential_smoothing(
	history: Union[List[float], np.ndarray],
	alpha: Optional[float] = None,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Compute Simple Exponential Smoothing (SES) forecast.

	If ``alpha`` is None, optimize alpha in [0.01, 0.99] to minimize MSE.
	"""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n < 1:
		raise ValueError("history must contain at least 1 observation")

	def _simulate_ses(a):
		fitted_vals = np.zeros(n, dtype=float)
		level = data[0]
		fitted_vals[0] = level
		for t in range(1, n):
			level = a * data[t - 1] + (1.0 - a) * level
			fitted_vals[t] = level
		return fitted_vals, level

	if alpha is None:
		if n == 1:
			alpha = 0.2
		else:
			def loss(a):
				f, _ = _simulate_ses(a[0])
				return float(np.mean((data[1:] - f[1:]) ** 2))
			res = minimize(loss, x0=[0.2], bounds=[(0.01, 0.99)], method='L-BFGS-B')
			alpha = float(res.x[0])
	else:
		if not (0 < alpha <= 1.0):
			raise ValueError("alpha must be in (0, 1]")

	fitted, final_level = _simulate_ses(alpha)
	next_forecast = alpha * data[-1] + (1.0 - alpha) * final_level if n > 0 else 0.0
	forecasts = np.full(forecast_periods, next_forecast, dtype=float)

	residuals = data - fitted
	metrics = evaluate_forecast(data[1:], fitted[1:]) if n > 1 else evaluate_forecast(data, fitted)

	return ForecastResult(
		method='SES',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={'alpha': round(alpha, 4)},
		metrics=metrics
	)


def holt_linear(
	history: Union[List[float], np.ndarray],
	alpha: Optional[float] = None,
	beta: Optional[float] = None,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Compute Holt's Linear Trend Model (Double Exponential Smoothing).

	If ``alpha`` or ``beta`` is None, optimizes parameters to minimize MSE.
	"""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n < 2:
		# Fall back to SES if series too short
		return simple_exponential_smoothing(data, alpha=0.2, forecast_periods=forecast_periods)

	def _simulate_holt(a, b):
		level = data[0]
		trend = data[1] - data[0]
		fitted_vals = np.zeros(n, dtype=float)
		fitted_vals[0] = level
		fitted_vals[1] = level + trend

		for t in range(2, n):
			val = data[t - 1]
			last_level = level
			level = a * val + (1.0 - a) * (last_level + trend)
			trend = b * (level - last_level) + (1.0 - b) * trend
			fitted_vals[t] = level + trend

		# Final update including data[-1]
		last_level = level
		level = a * data[-1] + (1.0 - a) * (last_level + trend)
		trend = b * (level - last_level) + (1.0 - b) * trend
		return fitted_vals, level, trend

	if alpha is None or beta is None:
		def loss(params):
			f, _, _ = _simulate_holt(params[0], params[1])
			return float(np.mean((data[1:] - f[1:]) ** 2))
		res = minimize(loss, x0=[0.3, 0.1], bounds=[(0.01, 0.99), (0.01, 0.99)], method='L-BFGS-B')
		alpha = float(res.x[0])
		beta = float(res.x[1])

	fitted, end_level, end_trend = _simulate_holt(alpha, beta)
	forecasts = np.array([end_level + (h + 1) * end_trend for h in range(forecast_periods)], dtype=float)
	# Non-negative demand constraint for inventory
	forecasts = np.maximum(0.0, forecasts)

	residuals = data - fitted
	metrics = evaluate_forecast(data[1:], fitted[1:])

	return ForecastResult(
		method='HoltLinear',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={'alpha': round(alpha, 4), 'beta': round(beta, 4)},
		metrics=metrics
	)


def holt_winters(
	history: Union[List[float], np.ndarray],
	seasonal_periods: int = 7,
	alpha: Optional[float] = None,
	beta: Optional[float] = None,
	gamma: Optional[float] = None,
	seasonal_type: str = 'add',
	forecast_periods: int = 1
) -> ForecastResult:
	"""Compute Holt-Winters Exponential Smoothing with Seasonality (Additive or Multiplicative)."""
	data = np.asarray(history, dtype=float)
	n = len(data)
	m = seasonal_periods

	if n < 2 * m:
		# Fall back to Holt's linear if less than two full seasonal cycles
		return holt_linear(data, alpha=alpha, beta=beta, forecast_periods=forecast_periods)

	is_mult = seasonal_type.lower() in ('mul', 'multiplicative')

	def _simulate_hw(a, b, g):
		# Initial level and trend
		level = float(np.mean(data[:m]))
		trend = float(np.mean(data[m:2*m] - data[:m])) / m

		# Initial seasonal indices
		seasonals = np.zeros(m, dtype=float)
		if is_mult:
			seasonals = np.maximum(1e-4, data[:m] / (level or 1.0))
			norm_factor = np.mean(seasonals)
			if norm_factor > 0:
				seasonals /= norm_factor
		else:
			seasonals = data[:m] - level

		fitted_vals = np.zeros(n, dtype=float)
		for t in range(m):
			fitted_vals[t] = level + seasonals[t] if not is_mult else level * seasonals[t]

		curr_seasonals = seasonals.copy()
		for t in range(m, n):
			s_idx = t % m
			val = data[t - 1]
			last_level = level
			s_prev = curr_seasonals[s_idx]

			if is_mult:
				s_safe = s_prev if abs(s_prev) > 1e-4 else 1e-4
				level = a * (val / s_safe) + (1.0 - a) * (last_level + trend)
				trend = b * (level - last_level) + (1.0 - b) * trend
				curr_seasonals[s_idx] = g * (val / (level or 1.0)) + (1.0 - g) * s_prev
				fitted_vals[t] = (level + trend) * curr_seasonals[s_idx]
			else:
				level = a * (val - s_prev) + (1.0 - a) * (last_level + trend)
				trend = b * (level - last_level) + (1.0 - b) * trend
				curr_seasonals[s_idx] = g * (val - level) + (1.0 - g) * s_prev
				fitted_vals[t] = (level + trend) + curr_seasonals[s_idx]

		# Final update for data[-1]
		s_idx = (n - 1) % m
		val = data[-1]
		last_level = level
		s_prev = curr_seasonals[s_idx]
		if is_mult:
			s_safe = s_prev if abs(s_prev) > 1e-4 else 1e-4
			level = a * (val / s_safe) + (1.0 - a) * (last_level + trend)
			trend = b * (level - last_level) + (1.0 - b) * trend
			curr_seasonals[s_idx] = g * (val / (level or 1.0)) + (1.0 - g) * s_prev
		else:
			level = a * (val - s_prev) + (1.0 - a) * (last_level + trend)
			trend = b * (level - last_level) + (1.0 - b) * trend
			curr_seasonals[s_idx] = g * (val - level) + (1.0 - g) * s_prev

		return fitted_vals, level, trend, curr_seasonals

	if alpha is None or beta is None or gamma is None:
		def loss(params):
			f, _, _, _ = _simulate_hw(params[0], params[1], params[2])
			return float(np.mean((data[m:] - f[m:]) ** 2))
		res = minimize(loss, x0=[0.2, 0.1, 0.1], bounds=[(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)], method='L-BFGS-B')
		alpha, beta, gamma = float(res.x[0]), float(res.x[1]), float(res.x[2])

	fitted, end_level, end_trend, end_seasonals = _simulate_hw(alpha, beta, gamma)
	forecasts = np.zeros(forecast_periods, dtype=float)
	for h in range(forecast_periods):
		s_idx = (n + h) % m
		if is_mult:
			forecasts[h] = max(0.0, (end_level + (h + 1) * end_trend) * end_seasonals[s_idx])
		else:
			forecasts[h] = max(0.0, (end_level + (h + 1) * end_trend) + end_seasonals[s_idx])

	residuals = data - fitted
	metrics = evaluate_forecast(data[m:], fitted[m:])

	return ForecastResult(
		method='HoltWinters',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={
			'alpha': round(alpha, 4),
			'beta': round(beta, 4),
			'gamma': round(gamma, 4),
			'seasonal_periods': m,
			'seasonal_type': seasonal_type
		},
		metrics=metrics
	)


def croston(
	history: Union[List[float], np.ndarray],
	alpha: float = 0.1,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Croston's Method for Intermittent / Slow-Moving Demand.

	Decomposes intermittent demand into demand size (when demand > 0)
	and inter-arrival intervals between positive demand periods.
	"""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n < 1:
		raise ValueError("history must contain at least 1 observation")

	# Find first non-zero demand
	non_zero_indices = np.where(data > 0)[0]
	if len(non_zero_indices) == 0:
		# Entire history is zeros
		return ForecastResult(
			method='Croston',
			forecasts=np.zeros(forecast_periods, dtype=float),
			fitted_values=np.zeros(n, dtype=float),
			residuals=np.zeros(n, dtype=float),
			parameters={'alpha': alpha},
			metrics={'mae': 0.0, 'rmse': 0.0, 'wape': 0.0, 'mape': 0.0, 'smape': 0.0}
		)

	first_nz = non_zero_indices[0]
	z_hat = float(data[first_nz])  # estimated demand size
	p_hat = 1.0                    # estimated inter-arrival interval
	q = 1                          # intervals since last demand

	fitted = np.zeros(n, dtype=float)
	fitted[:first_nz + 1] = z_hat / p_hat

	for t in range(first_nz + 1, n):
		if data[t] > 0:
			z_hat = alpha * data[t] + (1.0 - alpha) * z_hat
			p_hat = alpha * q + (1.0 - alpha) * p_hat
			q = 1
		else:
			q += 1
		fitted[t] = z_hat / (p_hat or 1.0)

	forecast_val = max(0.0, z_hat / (p_hat or 1.0))
	forecasts = np.full(forecast_periods, forecast_val, dtype=float)
	residuals = data - fitted
	metrics = evaluate_forecast(data, fitted)

	return ForecastResult(
		method='Croston',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={'alpha': alpha, 'final_demand_size': round(z_hat, 4), 'final_interval': round(p_hat, 4)},
		metrics=metrics
	)


def sba(
	history: Union[List[float], np.ndarray],
	alpha: float = 0.1,
	forecast_periods: int = 1
) -> ForecastResult:
	"""Syntetos-Boylan Approximation (SBA) for Intermittent Demand.

	Multiplies Croston's estimate by (1 - alpha/2) to remove the positive bias
	inherent in Croston's division of expected size by expected interval.
	"""
	croston_res = croston(history, alpha=alpha, forecast_periods=forecast_periods)
	correction = (1.0 - alpha / 2.0)
	forecasts = np.maximum(0.0, croston_res.forecasts * correction)
	fitted = np.maximum(0.0, croston_res.fitted_values * correction)
	residuals = np.asarray(history, dtype=float) - fitted
	metrics = evaluate_forecast(history, fitted)

	return ForecastResult(
		method='SBA',
		forecasts=forecasts,
		fitted_values=fitted,
		residuals=residuals,
		parameters={**croston_res.parameters, 'correction_factor': round(correction, 4)},
		metrics=metrics
	)


def auto_forecast(
	history: Union[List[float], np.ndarray],
	forecast_periods: int = 1,
	seasonal_periods: int = 7
) -> ForecastResult:
	"""Automatically select the best forecasting model minimizing WAPE on historical data.

	If demand is highly intermittent (> 40% zeros), considers Croston and SBA.
	Otherwise evaluates SMA, SES, Holt, and Holt-Winters (if data length permits).
	"""
	data = np.asarray(history, dtype=float)
	n = len(data)
	if n == 0:
		raise ValueError("history is empty")

	zero_ratio = float(np.sum(data == 0)) / n
	candidates: List[ForecastResult] = []

	if zero_ratio > 0.35:
		candidates.append(sba(data, alpha=0.1, forecast_periods=forecast_periods))
		candidates.append(croston(data, alpha=0.1, forecast_periods=forecast_periods))
		candidates.append(simple_exponential_smoothing(data, alpha=0.1, forecast_periods=forecast_periods))
	else:
		candidates.append(simple_moving_average(data, window=min(3, n), forecast_periods=forecast_periods))
		candidates.append(simple_exponential_smoothing(data, forecast_periods=forecast_periods))
		if n >= 3:
			candidates.append(holt_linear(data, forecast_periods=forecast_periods))
		if n >= 2 * seasonal_periods:
			try:
				candidates.append(holt_winters(data, seasonal_periods=seasonal_periods, forecast_periods=forecast_periods))
			except Exception:
				pass

	# Select candidate with lowest WAPE (or RMSE as tiebreaker)
	best = min(candidates, key=lambda c: (c.metrics.get('wape', 1e9), c.metrics.get('rmse', 1e9)))
	return best


# ===============================================================================
# Forecast-to-Inventory Bridge
# ===============================================================================

@dataclass
class InventoryParameters:
	"""Inventory policy parameters derived from demand forecasting."""
	mean_demand_per_period: float
	lead_time: float
	service_level: float
	lead_time_demand_mean: float
	lead_time_demand_std: float
	safety_factor_z: float
	safety_stock: float
	reorder_point: float
	base_stock_level: float
	economic_order_quantity: Optional[float] = None
	holding_cost: Optional[float] = None
	fixed_cost: Optional[float] = None

	def to_policy(self, policy_type: str = 'BS', **kwargs) -> Policy:
		"""Create a stockpyl Policy instance with the computed parameters."""
		ptype = policy_type.upper()
		if ptype == 'BS':
			return Policy(type='BS', base_stock_level=self.base_stock_level, **kwargs)
		elif ptype in ('RQ', 'R,Q'):
			q = self.economic_order_quantity if self.economic_order_quantity is not None else max(1.0, self.mean_demand_per_period)
			return Policy(type='rQ', reorder_point=self.reorder_point, order_quantity=q, **kwargs)
		elif ptype in ('SS', 'S,S'):
			q = self.economic_order_quantity if self.economic_order_quantity is not None else max(1.0, self.mean_demand_per_period)
			s_low = self.reorder_point
			s_high = s_low + q
			return Policy(type='sS', reorder_point=s_low, order_up_to_level=s_high, **kwargs)
		else:
			return Policy(type='BS', base_stock_level=self.base_stock_level, **kwargs)


def forecast_to_inventory_params(
	mean_demand: float,
	residual_std: float,
	lead_time: float = 1.0,
	service_level: float = 0.95,
	holding_cost: Optional[float] = None,
	fixed_cost: Optional[float] = None
) -> InventoryParameters:
	"""Compute safety stock, reorder point (r), base-stock level (S), and EOQ (Q)
	from forecast mean and forecast error standard deviation.

	Parameters
	----------
	mean_demand : float
		Estimated average periodic demand :math:`\\hat{\\mu}`.
	residual_std : float
		Standard deviation of forecast error :math:`\\hat{\\sigma}_e`.
	lead_time : float, optional
		Replenishment lead time in periods (default 1.0).
	service_level : float, optional
		Target non-stockout cycle service level, e.g. 0.95 (default 0.95).
	holding_cost : float, optional
		Unit holding cost per period :math:`h`.
	fixed_cost : float, optional
		Fixed order setup cost :math:`K`.

	Returns
	-------
	InventoryParameters
		Dataclass containing all derived replenishment parameters.
	"""
	if mean_demand < 0:
		raise ValueError("mean_demand must be non-negative")
	if residual_std < 0:
		raise ValueError("residual_std must be non-negative")
	if not (0 < service_level < 1.0):
		raise ValueError("service_level must be strictly between 0 and 1")

	ltd_mean = mean_demand * lead_time
	ltd_std = residual_std * math.sqrt(max(1.0, lead_time))
	z = float(stats.norm.ppf(service_level))
	safety_stock = max(0.0, z * ltd_std)
	reorder_point = ltd_mean + safety_stock
	base_stock_level = reorder_point

	eoq = None
	if holding_cost is not None and fixed_cost is not None and holding_cost > 0 and fixed_cost >= 0:
		eoq = math.sqrt(2.0 * mean_demand * fixed_cost / holding_cost) if mean_demand > 0 else 0.0

	return InventoryParameters(
		mean_demand_per_period=mean_demand,
		lead_time=lead_time,
		service_level=service_level,
		lead_time_demand_mean=ltd_mean,
		lead_time_demand_std=ltd_std,
		safety_factor_z=z,
		safety_stock=safety_stock,
		reorder_point=reorder_point,
		base_stock_level=base_stock_level,
		economic_order_quantity=eoq,
		holding_cost=holding_cost,
		fixed_cost=fixed_cost
	)
