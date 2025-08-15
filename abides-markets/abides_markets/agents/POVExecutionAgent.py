# POVExecutionAgent.py
import sys
import warnings
import pandas as pd
from .trading_agent import TradingAgent  # Corrected import path
import logging  # NEW: Use standard logging

logger = logging.getLogger(__name__)  # NEW: Logger for this module

POVExecutionWarning_msg = (
	"Running a configuration using POVExecutionAgent requires an ExchangeAgent with "
	"attribute `stream_history` set to a large value, recommended at sys.maxsize."
)


class POVExecutionAgent(TradingAgent):
	def __init__(
		self,
		id,
		name,
		type,
		symbol,
		starting_cash,
		direction,
		quantity,
		pov,
		start_time,
		freq,
		lookback_period,
		end_time=None,
		trade=True,
		log_orders=False,
		random_state=None,
	):
		super().__init__(
			id,
			name,
			type,
			starting_cash=starting_cash,
			log_orders=log_orders,
			random_state=random_state,
		)
		self.log_events = True  # save events for plotting
		self.symbol = symbol
		self.direction = direction
		self.quantity = quantity
		self.rem_quantity = quantity
		self.pov = pov
		self.start_time = start_time
		self.end_time = end_time
		self.freq = freq
		self.look_back_period = lookback_period
		self.trade = trade
		self.accepted_orders = []
		self.state = "AWAITING_WAKEUP"
		warnings.warn(POVExecutionWarning_msg, UserWarning, stacklevel=1)
		self.processEndTime()

	def processEndTime(self):
		"""Make end time of POV order sensible, i.e. if a time is given leave it alone; else, add 24 hours to start."""
		if self.end_time is None:
			self.end_time = self.start_time + pd.to_timedelta("24 hours").value

	def wakeup(self, currentTime):
		can_trade = super().wakeup(currentTime)
		# Now safe: this calls the new alias, which delegates to getWakeFrequency()
		self.set_wakeup(currentTime + self.get_wake_frequency())
		if not can_trade:
			return
		if (
			self.trade
			and self.rem_quantity > 0
			and self.start_time < currentTime < self.end_time
		):
			self.cancelOrders()
			self.getCurrentSpread(self.symbol, depth=sys.maxsize)
			self.get_transacted_volume(
				self.symbol, lookback_period=self.look_back_period
			)
			self.state = "AWAITING_TRANSACTED_VOLUME"

	# Canonical snake_case API (matches the rest of the codebase)
	def get_wake_frequency(self):
		return pd.Timedelta(self.freq).value

	# Optional: short-lived alias with a deprecation warning; remove once callers are clean.
	def getWakeFrequency(self):  # pragma: no cover
		import warnings
		warnings.warn(
			"getWakeFrequency() is deprecated; use get_wake_frequency().",
			DeprecationWarning,
			stacklevel=2,
		)
		return self.get_wake_frequency()

	def receiveMessage(self, currentTime, msg):
		super().receiveMessage(currentTime, msg)
		if msg.body["msg"] == "ORDER_EXECUTED":
			self.handleOrderExecution(currentTime, msg)
		elif msg.body["msg"] == "ORDER_ACCEPTED":
			self.handleOrderAcceptance(currentTime, msg)
		if currentTime > self.end_time:
			logger.info(  # CHANGED to logger.info
				f"[---- {self.name} - {currentTime} ----]: current time {currentTime} is after specified end time of POV order "
				f"{self.end_time}. TRADING CONCLUDED. "
			)
			return
		if (
			self.rem_quantity > 0
			and self.state == "AWAITING_TRANSACTED_VOLUME"
			and msg.body["msg"] == "QUERY_TRANSACTED_VOLUME"
			and self.transacted_volume[self.symbol] is not None
			and currentTime > self.start_time
		):
			qty = round(self.pov * self.transacted_volume[self.symbol])
			self.cancelOrders()
			self.placeMarketOrder(self.symbol, qty, self.direction == "BUY")
			logger.info(
				f"[---- {self.name} - {currentTime} ----]: TOTAL TRANSACTED VOLUME IN THE LAST {self.look_back_period} = {self.transacted_volume[self.symbol]}"
			)  # CHANGED
			logger.info(
				f"[---- {self.name} - {currentTime} ----]: MARKET ORDER PLACED - {qty}"
			)  # CHANGED

	def handleOrderAcceptance(self, currentTime, msg):
		accepted_order = msg.body["order"]
		self.accepted_orders.append(accepted_order)
		accepted_qty = sum(
			accepted_order.quantity for accepted_order in self.accepted_orders
		)
		logger.info(
			f"[---- {self.name} - {currentTime} ----]: ACCEPTED QUANTITY : {accepted_qty}"
		)  # CHANGED

	def handleOrderExecution(self, currentTime, msg):
		executed_order = msg.body["order"]
		self.executed_orders.append(executed_order)
		executed_qty = sum(
			executed_order.quantity for executed_order in self.executed_orders
		)
		self.rem_quantity = self.quantity - executed_qty
		logger.info(
			f"[---- {self.name} - {currentTime} ----]: LIMIT ORDER EXECUTED - {executed_order.quantity} @ {executed_order.fill_price}"
		)  # CHANGED
		logger.info(
			f"[---- {self.name} - {currentTime} ----]: EXECUTED QUANTITY: {executed_qty}"
		)  # CHANGED
		logger.info(
			f"[---- {self.name} - {currentTime} ----]: REMAINING QUANTITY (NOT EXECUTED): {self.rem_quantity}"
		)  # CHANGED
		logger.info(
			f"[---- {self.name} - {currentTime} ----]: % EXECUTED: {round((1 - self.rem_quantity / self.quantity) * 100, 2)} \n"
		)  # CHANGED

	def cancelOrders(self):
		for _, order in self.orders.items():
			self.cancelOrder(order)
