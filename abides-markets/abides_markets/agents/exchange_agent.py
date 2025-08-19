import datetime as dt
import logging
import warnings
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from abides_core import Kernel, Message, NanosecondTime
from ..messages.market import (
	MarketClosedMsg,
	MarketHoursMsg,
	MarketHoursRequestMsg,
	MarketClosePriceRequestMsg,
	MarketClosePriceMsg,
)
from ..messages.marketdata import (
	BookImbalanceDataMsg,
	L1SubReqMsg,
	L2SubReqMsg,
	L3SubReqMsg,
	TransactedVolSubReqMsg,
	MarketDataSubReqMsg,
	L1DataMsg,
	L2DataMsg,
	L3DataMsg,
	TransactedVolDataMsg,
	BookImbalanceSubReqMsg,
	MarketDataEventMsg,
)
from ..messages.order import (
	LimitOrderMsg,
	MarketOrderMsg,
	PartialCancelOrderMsg,
	CancelOrderMsg,
	ModifyOrderMsg,
	ReplaceOrderMsg,
	OrderMsg,
)
from ..messages.orderbook import OrderAcceptedMsg, OrderExecutedMsg, OrderCancelledMsg
from ..messages.query import (
	QueryLastTradeMsg,
	QueryLastTradeResponseMsg,
	QueryMsg,
	QuerySpreadMsg,
	QuerySpreadResponseMsg,
	QueryOrderStreamMsg,
	QueryOrderStreamResponseMsg,
	QueryTransactedVolMsg,
	QueryTransactedVolResponseMsg,
)
from ..orders import Side
from ..order_book import OrderBook
from .financial_agent import FinancialAgent

logger = logging.getLogger(__name__)
pd.set_option("display.max_rows", 500)

class ExchangeAgent(FinancialAgent):
	"""
	The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp
	to open and close trading, a list of equity symbols for which it should create order
	books, a frequency at which to archive snapshots of its order books, a pipeline
	delay (in ns) for order activity, the exchange computation delay (in ns), the levels
	of order stream history to maintain per symbol (maintains all orders that led to the
	last N trades), whether to log all order activity to the agent log, and a random
	state object (already seeded) to use for stochasticity.
	"""
	@dataclass
	class MetricTracker(ABC):
		total_time_no_liquidity_asks: int = 0
		total_time_no_liquidity_bids: int = 0
		pct_time_no_liquidity_asks: float = 0
		pct_time_no_liquidity_bids: float = 0
		total_exchanged_volume: int = 0
		last_trade: Optional[int] = 0

	@dataclass
	class BaseDataSubscription(ABC):
		agent_id: int
		last_update_ts: int

	@dataclass
	class FrequencyBasedSubscription(BaseDataSubscription, ABC):
		freq: int

	@dataclass
	class L1DataSubscription(FrequencyBasedSubscription):
		pass

	@dataclass
	class L2DataSubscription(FrequencyBasedSubscription):
		depth: int

	@dataclass
	class L3DataSubscription(FrequencyBasedSubscription):
		depth: int

	@dataclass
	class TransactedVolDataSubscription(FrequencyBasedSubscription):
		lookback: str

	@dataclass
	class EventBasedSubscription(BaseDataSubscription, ABC):
		event_in_progress: bool

	@dataclass
	class BookImbalanceDataSubscription(EventBasedSubscription):
		min_imbalance: float
		imbalance: Optional[float] = None
		side: Optional[Side] = None

	def __init__(
		self,
		id: int,
		mkt_open: NanosecondTime,
		mkt_close: NanosecondTime,
		symbols: List[str],
		name: Optional[str] = None,
		type: Optional[str] = None,
		random_state: Optional[np.random.RandomState] = None,
		book_logging: bool = True,
		book_log_depth: int = 10,
		pipeline_delay: int = 40000,
		computation_delay: int = 1,
		stream_history: int = 0,
		log_orders: bool = False,
		use_metric_tracker: bool = True,
	) -> None:
		super().__init__(id, name, type, random_state)
		self.symbols = symbols
		self.reschedule: bool = False
		self.mkt_open: NanosecondTime = mkt_open
		self.mkt_close: NanosecondTime = mkt_close
		self.pipeline_delay: int = pipeline_delay
		self.computation_delay: int = computation_delay
		self.stream_history: int = stream_history
		self.book_logging: bool = book_logging
		self.book_log_depth: int = book_log_depth
		self.log_orders: bool = log_orders
		self.order_books: Dict[str, OrderBook] = {
			symbol: OrderBook(self, symbol) for symbol in symbols
		}
		if use_metric_tracker:
			self.metric_trackers: Dict[str, ExchangeAgent.MetricTracker] = {
				symbol: self.MetricTracker() for symbol in symbols
			}
		self.data_subscriptions: DefaultDict[
			str, List[ExchangeAgent.BaseDataSubscription]
		] = defaultdict(list)
		self.market_close_price_subscriptions: List[int] = []

	def kernel_initializing(self, kernel: "Kernel") -> None:
		super().kernel_initializing(kernel)
		assert self.kernel is not None
		self.oracle = self.kernel.oracle
		for symbol in self.order_books:
			try:
				self.order_books[symbol].last_trade = self.oracle.get_daily_open_price(
					symbol, self.mkt_open
				)
				logger.debug(
					"Opening price for {} is {}".format(
						symbol, self.order_books[symbol].last_trade
					)
				)
			except AttributeError as e:
				logger.debug(str(e))

		self.set_wakeup(self.mkt_close)

	def kernel_terminating(self) -> None:
		super().kernel_terminating()
		for symbol in self.symbols:
			self.analyse_order_book(symbol)
			bid_volume, ask_volume = self.order_books[symbol].get_transacted_volume(
				self.current_time - self.mkt_open
			)
			self.metric_trackers[symbol].total_exchanged_volume = (
				bid_volume + ask_volume
			)
			self.metric_trackers[symbol].last_trade = self.order_books[
				symbol
			].last_trade
		if self.log_orders == None:
			return
		if hasattr(self.oracle, "f_log"):
			for symbol in self.oracle.f_log:
				dfFund = pd.DataFrame(self.oracle.f_log[symbol])
				if not dfFund.empty:
					dfFund.set_index("FundamentalTime", inplace=True)
					self.write_log(dfFund, filename="fundamental_{}".format(symbol))
					logger.debug("Fundamental archival complete.")

	def wakeup(self, current_time: NanosecondTime):
		super().wakeup(current_time)
		if current_time >= self.mkt_close:
			message = MarketClosePriceMsg(
				{symbol: book.last_trade for symbol, book in self.order_books.items()}
			)
			for agent in self.market_close_price_subscriptions:
				self.send_message(agent, message)

	def receive_message(
		self, current_time: NanosecondTime, sender_id: int, message: Message
	) -> None:
		super().receive_message(current_time, sender_id, message)
		self.set_computation_delay(self.computation_delay)
		if current_time > self.mkt_close:
			if isinstance(message, OrderMsg):
				if isinstance(message, ModifyOrderMsg):
					logger.debug(
						"{} received {}: OLD: {} NEW: {}".format(
							self.name,
							message.type(),
							message.old_order,
							message.new_order,
						)
					)
				else:
					logger.debug(
						"{} received {}: {}".format(
							self.name, message.type(), message.order
						)
					)
				self.send_message(sender_id, MarketClosedMsg())
				return
			elif isinstance(message, QueryMsg):
				pass
			else:
				logger.debug(
					"{} received {}, discarded: market is closed.".format(
						self.name, message.type()
					)
				)
				self.send_message(sender_id, MarketClosedMsg())
				return

		if isinstance(message, OrderMsg):
			if self.log_orders:
				if isinstance(message, (ModifyOrderMsg, ReplaceOrderMsg)):
					self.logEvent(
						message.type(),
						message.new_order.to_dict(),
						deepcopy_event=False,
					)
				else:
					self.logEvent(
						message.type(), message.order.to_dict(), deepcopy_event=False
					)
		else:
			self.logEvent(message.type(), message)

		if isinstance(message, MarketDataSubReqMsg):
			if message.symbol not in self.order_books:
				logger.warning(f"Subscription request discarded. Unknown symbol: {message.symbol}")
				return
			if message.cancel:
				logger.debug(
					"{} received MarketDataSubscriptionCancellation request from agent {}".format(
						self.name, sender_id
					)
				)
				for data_sub in self.data_subscriptions[message.symbol][:]:
					if (
						data_sub.agent_id == sender_id
						and data_sub.freq == message.freq
						and data_sub.depth == message.depth
						and data_sub.__class__ == message.__class__
					):
						self.data_subscriptions[message.symbol].remove(data_sub)
			else:
				logger.debug(
					"{} received MarketDataSubscriptionRequest from agent {}".format(
						self.name, sender_id
					)
				)
				if isinstance(message, L1SubReqMsg):
					sub = self.L1DataSubscription(sender_id, current_time, message.freq)
				elif isinstance(message, L2SubReqMsg):
					sub = self.L2DataSubscription(sender_id, current_time, message.freq, message.depth)
				elif isinstance(message, L3SubReqMsg):
					sub = self.L3DataSubscription(sender_id, current_time, message.freq, message.depth)
				elif isinstance(message, TransactedVolSubReqMsg):
					sub = self.TransactedVolDataSubscription(sender_id, current_time, message.freq, message.lookback)
				elif isinstance(message, BookImbalanceSubReqMsg):
					sub = self.BookImbalanceDataSubscription(sender_id, current_time, False, message.min_imbalance)
				else:
					raise Exception("Invalid subscription type")
				self.data_subscriptions[message.symbol].append(sub)
				# Log subscription details for debugging
				logger.debug(f"Registered subscription for agent {sender_id}, symbol {message.symbol}, freq {message.freq}, depth {message.depth}")

		if isinstance(message, MarketHoursRequestMsg):
			logger.debug(
				"{} received market hours request from agent {}".format(self.name, sender_id)
			)
			self.set_computation_delay(0)
			self.send_message(sender_id, MarketHoursMsg(self.mkt_open, self.mkt_close))

		elif isinstance(message, MarketClosePriceRequestMsg):
			self.market_close_price_subscriptions.append(sender_id)

		elif isinstance(message, QueryLastTradeMsg):
			symbol = message.symbol
			if symbol not in self.order_books:
				warnings.warn(f"Last trade request discarded. Unknown symbol: {symbol}")
			else:
				logger.debug(
					"{} received QUERY_LAST_TRADE ({}) request from agent {}".format(
						self.name, symbol, sender_id
					)
				)
				self.send_message(
					sender_id,
					QueryLastTradeResponseMsg(
						symbol=symbol,
						last_trade=self.order_books[symbol].last_trade,
						mkt_closed=current_time > self.mkt_close,
					),
				)

		elif isinstance(message, QuerySpreadMsg):
			symbol = message.symbol
			depth = message.depth
			if symbol not in self.order_books:
				warnings.warn(f"Bid-ask spread request discarded. Unknown symbol: {symbol}")
			else:
				logger.debug(
					"{} received QUERY_SPREAD ({}:{}) request from agent {}".format(
						self.name, symbol, depth, sender_id
					)
				)
				self.send_message(
					sender_id,
					QuerySpreadResponseMsg(
						symbol=symbol,
						depth=depth,
						bids=self.order_books[symbol].get_l2_bid_data(depth),
						asks=self.order_books[symbol].get_l2_ask_data(depth),
						last_trade=self.order_books[symbol].last_trade,
						mkt_closed=current_time > self.mkt_close,
					),
				)

		elif isinstance(message, QueryOrderStreamMsg):
			symbol = message.symbol
			length = message.length
			if symbol not in self.order_books:
				warnings.warn(f"Order stream request discarded. Unknown symbol: {symbol}")
			else:
				logger.debug(
					"{} received QUERY_ORDER_STREAM ({}:{}) request from agent {}".format(
						self.name, symbol, length, sender_id
					)
				)
				self.send_message(
					sender_id,
					QueryOrderStreamResponseMsg(
						symbol=symbol,
						length=length,
						orders=self.order_books[symbol].history[1 : length + 1],
						mkt_closed=current_time > self.mkt_close,
					),
				)

		elif isinstance(message, QueryTransactedVolMsg):
			symbol = message.symbol
			lookback_period = message.lookback_period
			if symbol not in self.order_books:
				warnings.warn(f"Order stream request discarded. Unknown symbol: {symbol}")
			else:
				logger.debug(
					"{} received QUERY_TRANSACTED_VOLUME ({}:{}) request from agent {}".format(
						self.name, symbol, lookback_period, sender_id
					)
				)
				bid_volume, ask_volume = self.order_books[symbol].get_transacted_volume(
					lookback_period
				)
				self.send_message(
					sender_id,
					QueryTransactedVolResponseMsg(
						symbol=symbol,
						bid_volume=bid_volume,
						ask_volume=ask_volume,
						mkt_closed=current_time > self.mkt_close,
					),
				)

		elif isinstance(message, LimitOrderMsg):
			logger.debug("{} received LIMIT_ORDER: {}".format(self.name, message.order))
			if message.order.symbol not in self.order_books:
				warnings.warn(f"Limit Order discarded. Unknown symbol: {message.order.symbol}")
			else:
				# Hand the order to the order book for processing
				order_book = self.order_books[message.order.symbol]
				order_book.handle_limit_order(deepcopy(message.order))
				# Log order acceptance and current LOB state
				logger.debug(
					"SENT: notifications of order acceptance to agent %s for order %s",
					message.order.agent_id,
					message.order.order_id
				)
				# Log L1 data for debugging
				bid_data = order_book.get_l1_bid_data()
				ask_data = order_book.get_l1_ask_data()
				logger.debug(
					f"LOB after limit order: symbol={message.order.symbol}, "
					f"best_bid={bid_data[0] if bid_data else None}, "
					f"best_ask={ask_data[0] if ask_data else None}, "
					f"bid_size={bid_data[1] if bid_data else None}, "
					f"ask_size={ask_data[1] if ask_data else None}"
				)
				self.publish_order_book_data()

		elif isinstance(message, MarketOrderMsg):
			logger.debug(
				"{} received MARKET_ORDER: {}".format(self.name, message.order)
			)
			if message.order.symbol not in self.order_books:
				warnings.warn(f"Market Order discarded. Unknown symbol: {message.order.symbol}")
			else:
				self.order_books[message.order.symbol].handle_market_order(
					deepcopy(message.order)
				)
				self.publish_order_book_data()

		elif isinstance(message, CancelOrderMsg):
			tag = message.tag
			metadata = message.metadata
			logger.debug(
				"{} received CANCEL_ORDER: {}".format(self.name, message.order)
			)
			if message.order.symbol not in self.order_books:
				warnings.warn(f"Cancellation request discarded. Unknown symbol: {message.order.symbol}")
			else:
				self.order_books[message.order.symbol].cancel_order(
					deepcopy(message.order), tag, metadata
				)
				self.publish_order_book_data()

		elif isinstance(message, PartialCancelOrderMsg):
			tag = message.tag
			metadata = message.metadata
			logger.debug(
				"{} received PARTIAL_CANCEL_ORDER: {}, new order: {}".format(
					self.name, message.order, message.quantity
				)
			)
			if message.order.symbol not in self.order_books:
				warnings.warn(f"Modification request discarded. Unknown symbol: {message.order.symbol}")
			else:
				self.order_books[message.order.symbol].partial_cancel_order(
					deepcopy(message.order), message.quantity, tag, metadata
				)
				self.publish_order_book_data()

		elif isinstance(message, ModifyOrderMsg):
			old_order = message.old_order
			new_order = message.new_order
			logger.debug(
				"{} received MODIFY_ORDER: {}, new order: {}".format(
					self.name, old_order, new_order
				)
			)
			if old_order.symbol not in self.order_books:
				warnings.warn(f"Modification request discarded. Unknown symbol: {old_order.symbol}")
			else:
				self.order_books[old_order.symbol].modify_order(
					deepcopy(old_order), deepcopy(new_order)
				)
				self.publish_order_book_data()

		elif isinstance(message, ReplaceOrderMsg):
			agent_id = message.agent_id
			order = message.old_order
			new_order = message.new_order
			logger.debug(
				"{} received REPLACE_ORDER: {}, new order: {}".format(
					self.name, order, new_order
				)
			)
			if order.symbol not in self.order_books:
				warnings.warn(f"Replacement request discarded. Unknown symbol: {order.symbol}")
			else:
				self.order_books[order.symbol].replace_order(
					agent_id, deepcopy(order), deepcopy(new_order)
				)
				self.publish_order_book_data()

	def publish_order_book_data(self) -> None:
		for symbol, data_subs in self.data_subscriptions.items():
			book = self.order_books[symbol]
			for data_sub in data_subs:
				if isinstance(data_sub, self.FrequencyBasedSubscription):
					messages = self.handle_frequency_based_data_subscription(
						symbol, data_sub
					)
				elif isinstance(data_sub, self.EventBasedSubscription):
					messages = self.handle_event_based_data_subscription(
						symbol, data_sub
					)
				else:
					raise Exception("Got invalid data subscription object")
				for message in messages:
					self.send_message(data_sub.agent_id, message)
					# Log sent market data for debugging
					logger.debug(
						f"Sent market data to agent {data_sub.agent_id}: "
						f"type={type(message).__name__}, "
						f"symbol={symbol}, "
						f"data={message.__dict__}"
					)
				if len(messages) > 0:
					data_sub.last_update_ts = book.last_update_ts

	def handle_frequency_based_data_subscription(
		self, symbol: str, data_sub: "ExchangeAgent.FrequencyBasedSubscription"
	) -> List[Message]:
		book = self.order_books[symbol]
		if (book.last_update_ts - data_sub.last_update_ts) < data_sub.freq:
			return []
		messages = []
		if isinstance(data_sub, self.L1DataSubscription):
			bid = book.get_l1_bid_data()
			ask = book.get_l1_ask_data()
			messages.append(
				L1DataMsg(symbol, book.last_trade, self.current_time, bid, ask)
			)
		elif isinstance(data_sub, self.L2DataSubscription):
			bids = book.get_l2_bid_data(data_sub.depth)
			asks = book.get_l2_ask_data(data_sub.depth)
			messages.append(
				L2DataMsg(
					symbol,
					book.last_trade,
					self.current_time,
					bids,
					asks,
				)
			)
		elif isinstance(data_sub, self.L3DataSubscription):
			bids = book.get_l3_bid_data(data_sub.depth)
			asks = book.get_l3_ask_data(data_sub.depth)
			messages.append(
				L3DataMsg(
					symbol,
					book.last_trade,
					self.current_time,
					bids,
					asks,
				)
			)
		elif isinstance(data_sub, self.TransactedVolDataSubscription):
			bid_volume, ask_volume = book.get_transacted_volume(data_sub.lookback)
			messages.append(
				TransactedVolDataMsg(
					symbol,
					book.last_trade,
					self.current_time,
					bid_volume,
					ask_volume,
				)
			)
		else:
			raise Exception("Got invalid data subscription object")
		return messages

	def handle_event_based_data_subscription(
		self, symbol: str, data_sub: "ExchangeAgent.EventBasedSubscription"
	) -> List[Message]:
		book = self.order_books[symbol]
		messages = []
		if isinstance(data_sub, self.BookImbalanceDataSubscription):
			imbalance, side = book.get_imbalance()
			event_in_progress = imbalance > data_sub.min_imbalance
			if data_sub.event_in_progress and event_in_progress:
				if side != data_sub.side:
					messages.append(
						BookImbalanceDataMsg(
							symbol,
							book.last_trade,
							self.current_time,
							MarketDataEventMsg.Stage.FINISH,
							data_sub.imbalance,
							data_sub.side,
						)
					)
					data_sub.event_in_progress = True
					data_sub.side = side
					data_sub.imbalance = imbalance
					messages.append(
						BookImbalanceDataMsg(
							symbol,
							book.last_trade,
							self.current_time,
							MarketDataEventMsg.Stage.START,
							imbalance,
							side,
						)
					)
			elif data_sub.event_in_progress and not event_in_progress:
				data_sub.event_in_progress = False
				data_sub.side = None
				data_sub.imbalance = None
				messages.append(
					BookImbalanceDataMsg(
						symbol,
						book.last_trade,
						self.current_time,
						MarketDataEventMsg.Stage.FINISH,
						imbalance,
						side,
					)
				)
			elif not data_sub.event_in_progress and event_in_progress:
				data_sub.event_in_progress = True
				data_sub.side = side
				data_sub.imbalance = imbalance
				messages.append(
					BookImbalanceDataMsg(
						symbol,
						book.last_trade,
						self.current_time,
						MarketDataEventMsg.Stage.START,
						imbalance,
						side,
					)
				)
		else:
			raise Exception("Got invalid data subscription object")
		return messages

	def logL2style(self, symbol: str) -> Optional[Tuple[List, List]]:
		book = self.order_books[symbol]
		if not book.book_log2:
			return None
		tmp = book.book_log2
		times = []
		booktop = []
		for t in tmp:
			times.append(t["QuoteTime"])
			booktop.append([t["bids"], t["asks"]])
		return (times, booktop)

	def send_message(self, recipient_id: int, message: Message) -> None:
		if isinstance(message, (OrderAcceptedMsg, OrderCancelledMsg, OrderExecutedMsg)):
			super().send_message(recipient_id, message, delay=self.pipeline_delay)
			if self.log_orders:
				self.logEvent(message.type(), message.order.to_dict())
		else:
			super().send_message(recipient_id, message)

	def analyse_order_book(self, symbol: str):
		book = self.order_books[symbol].book_log2
		self.get_time_dropout(book, symbol)

	def get_time_dropout(self, book: List[Dict[str, Any]], symbol: str):
		if len(book) == 0:
			return
		df = pd.DataFrame(book)
		total_time = df["QuoteTime"].iloc[-1] - df["QuoteTime"].iloc[0]
		is_null_bids = False
		t_null_bids_first = 0
		T_null_bids = 0
		is_null_asks = False
		t_null_asks_first = 0
		T_null_asks = 0
		for _, row in df.iterrows():
			if (len(row["bids"]) == 0) & (is_null_bids == False):
				t_null_bids_first = row["QuoteTime"]
				is_null_bids = True
			elif (len(row["bids"]) != 0) & (is_null_bids == True):
				T_null_bids += row["QuoteTime"] - t_null_bids_first
				is_null_bids = False
			if (len(row["asks"]) == 0) & (is_null_asks == False):
				t_null_asks_first = row["QuoteTime"]
				is_null_asks = True
			elif (len(row["asks"]) != 0) & (is_null_asks == True):
				T_null_asks += row["QuoteTime"] - t_null_asks_first
				is_null_asks = False
		self.metric_trackers[symbol] = self.MetricTracker(
			total_time_no_liquidity_asks=T_null_asks / 1e9,
			total_time_no_liquidity_bids=T_null_bids / 1e9,
			pct_time_no_liquidity_asks=100 * T_null_asks / total_time,
			pct_time_no_liquidity_bids=100 * T_null_bids / total_time,
		)
