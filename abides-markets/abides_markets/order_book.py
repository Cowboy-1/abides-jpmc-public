import logging
import sys
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from abides_core import Agent, NanosecondTime
from abides_core.utils import str_to_ns, ns_date
from .messages.orderbook import (
	OrderAcceptedMsg,
	OrderExecutedMsg,
	OrderCancelledMsg,
	OrderPartialCancelledMsg,
	OrderModifiedMsg,
	OrderReplacedMsg,
)
from .orders import LimitOrder, MarketOrder, Order, Side
from .price_level import PriceLevel

logger = logging.getLogger(__name__)

class OrderBook:
	"""Basic class for an order book for one symbol, in the style of the major US Stock Exchanges."""
	def __init__(self, owner: Agent, symbol: str) -> None:
		self.owner: Agent = owner
		self.symbol: str = symbol
		self.bids: List[PriceLevel] = []
		self.asks: List[PriceLevel] = []
		self.last_trade: Optional[int] = None
		self.book_log2: List[Dict[str, Any]] = []
		self.quotes_seen: Set[int] = set()
		self.history: List[Dict[str, Any]] = []
		self.last_update_ts: Optional[NanosecondTime] = self.owner.mkt_open
		self.buy_transactions: List[Tuple[NanosecondTime, int]] = []
		self.sell_transactions: List[Tuple[NanosecondTime, int]] = []

	def handle_limit_order(self, order: LimitOrder, quiet: bool = False) -> None:
		if order.symbol != self.symbol:
			warnings.warn(f"{order.symbol} order discarded. Does not match OrderBook symbol: {self.symbol}")
			return
		if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
			warnings.warn(f"{order.symbol} order discarded. Quantity ({order.quantity}) must be a positive integer.")
			return
		if (order.limit_price < 0) or (int(order.limit_price) != order.limit_price):
			warnings.warn(f"{order.symbol} order discarded. Limit price ({order.limit_price}) must be a positive integer.")
			return
		executed: List[Tuple[int, int]] = []
		while True:
			matched_order = self.execute_order(order)
			if matched_order is not None:
				assert matched_order.fill_price is not None
				executed.append((matched_order.quantity, matched_order.fill_price))
				if order.quantity <= 0:
					break
			else:
				self.enter_order(deepcopy(order), quiet=quiet)
				logger.debug("ACCEPTED: new order %s", order)
				logger.debug(
					"SENT: notifications of order acceptance to agent %s for order %s",
					order.agent_id,
					order.order_id
				)  # Fixed: Use %s formatting
				if not quiet:
					self.owner.send_message(order.agent_id, OrderAcceptedMsg(order))
				# Log LOB state after order acceptance
				bid_data = self.get_l1_bid_data()
				ask_data = self.get_l1_ask_data()
				logger.debug(
					f"LOB after limit order acceptance: symbol={self.symbol}, "
					f"best_bid={bid_data[0] if bid_data else None}, "
					f"best_ask={ask_data[0] if ask_data else None}, "
					f"bid_size={bid_data[1] if bid_data else None}, "
					f"ask_size={ask_data[1] if ask_data else None}"
				)
				break
		if self.bids:
			self.owner.logEvent(
				"BEST_BID",
				"{},{},{}".format(self.symbol, self.bids[0].price, self.bids[0].total_quantity),
			)
		if self.asks:
			self.owner.logEvent(
				"BEST_ASK",
				"{},{},{}".format(self.symbol, self.asks[0].price, self.asks[0].total_quantity),
			)
		if len(executed) > 0:
			trade_qty = 0
			trade_price = 0
			for q, p in executed:
				logger.debug("Executed: %s @ %s", q, p)
				trade_qty += q
				trade_price += p * q
			avg_price = int(round(trade_price / trade_qty))
			logger.debug(f"Avg: {trade_qty} @ ${avg_price:0.4f}")
			self.owner.logEvent("LAST_TRADE", f"{trade_qty},${avg_price:0.4f}")
			self.last_trade = avg_price
			# Log LOB state after execution
			bid_data = self.get_l1_bid_data()
			ask_data = self.get_l1_ask_data()
			logger.debug(
				f"LOB after limit order execution: symbol={self.symbol}, "
				f"best_bid={bid_data[0] if bid_data else None}, "
				f"best_ask={ask_data[0] if ask_data else None}, "
				f"bid_size={bid_data[1] if bid_data else None}, "
				f"ask_size={ask_data[1] if ask_data else None}"
			)

	def handle_market_order(self, order: MarketOrder) -> None:
		if order.symbol != self.symbol:
			warnings.warn(f"{order.symbol} order discarded. Does not match OrderBook symbol: {self.symbol}")
			return
		if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
			warnings.warn(f"{order.symbol} order discarded. Quantity ({order.quantity}) must be a positive integer.")
			return
		order = deepcopy(order)
		while order.quantity > 0:
			if self.execute_order(order) is None:
				break
		# Log LOB state after market order
		bid_data = self.get_l1_bid_data()
		ask_data = self.get_l1_ask_data()
		logger.debug(
			f"LOB after market order: symbol={self.symbol}, "
			f"best_bid={bid_data[0] if bid_data else None}, "
			f"best_ask={ask_data[0] if ask_data else None}, "
			f"bid_size={bid_data[1] if bid_data else None}, "
			f"ask_size={ask_data[1] if ask_data else None}"
		)

	def execute_order(self, order: Order) -> Optional[Order]:
		book = self.asks if order.side.is_bid() else self.bids
		if len(book) == 0:
			return None
		elif isinstance(order, LimitOrder) and not book[0].order_is_match(order):
			return None
		elif order.tag in ["MR_preprocess_ADD", "MR_preprocess_REPLACE"]:
			self.owner.logEvent(order.tag + "_POST_ONLY", {"order_id": order.order_id})
			return None
		else:
			is_ptc_exec = False
			if order.quantity >= book[0].peek()[0].quantity:
				matched_order, matched_order_metadata = book[0].pop()
				if matched_order.is_price_to_comply:
					is_ptc_exec = True
					if matched_order_metadata["ptc_hidden"] == False:
						raise Exception("Should not be executing on the visible half of a price to comply order!")
					assert book[1].remove_order(matched_order.order_id) is not None
					if book[1].is_empty:
						del book[1]
				if book[0].is_empty:
					del book[0]
			else:
				book_order, book_order_metadata = book[0].peek()
				matched_order = deepcopy(book_order)
				matched_order.quantity = order.quantity
				book_order.quantity -= matched_order.quantity
				if book_order.is_price_to_comply:
					is_ptc_exec = True
					if book_order_metadata["ptc_hidden"] == False:
						raise Exception("Should not be executing on the visible half of a price to comply order!")
					book_order_metadata["ptc_other_half"].quantity -= matched_order.quantity
			matched_order.fill_price = matched_order.limit_price
			if order.side.is_bid():
				self.buy_transactions.append((self.owner.current_time, matched_order.quantity))
			else:
				self.sell_transactions.append((self.owner.current_time, matched_order.quantity))
			self.history.append(
				dict(
					time=self.owner.current_time,
					type="EXEC",
					order_id=matched_order.order_id,
					agent_id=matched_order.agent_id,
					oppos_order_id=order.order_id,
					oppos_agent_id=order.agent_id,
					side="SELL" if order.side.is_bid() else "BUY",
					quantity=matched_order.quantity,
					price=matched_order.limit_price if is_ptc_exec else None,
				)
			)
			filled_order = deepcopy(order)
			filled_order.quantity = matched_order.quantity
			filled_order.fill_price = matched_order.fill_price
			order.quantity -= filled_order.quantity
			logger.debug("MATCHED: new order %s vs old order %s", filled_order, matched_order)
			logger.debug(
				"SENT: notifications of order execution to agents %s and %s for orders %s and %s",
				filled_order.agent_id,
				matched_order.agent_id,
				filled_order.order_id,
				matched_order.order_id
			)
			self.owner.send_message(matched_order.agent_id, OrderExecutedMsg(matched_order))
			self.owner.send_message(order.agent_id, OrderExecutedMsg(filled_order))
			if self.owner.book_logging == True:
				self.append_book_log2()
			# Log LOB state after execution
			bid_data = self.get_l1_bid_data()
			ask_data = self.get_l1_ask_data()
			logger.debug(
				f"LOB after order execution: symbol={self.symbol}, "
				f"best_bid={bid_data[0] if bid_data else None}, "
				f"best_ask={ask_data[0] if ask_data else None}, "
				f"bid_size={bid_data[1] if bid_data else None}, "
				f"ask_size={ask_data[1] if ask_data else None}"
			)
			return matched_order

	def enter_order(
		self,
		order: LimitOrder,
		metadata: Optional[Dict] = None,
		quiet: bool = False,
	) -> None:
		if order.is_price_to_comply and (
			(metadata is None) or (metadata == {}) or ("ptc_hidden" not in metadata)
		):
			hidden_order = deepcopy(order)
			visible_order = deepcopy(order)
			hidden_order.is_hidden = True
			hidden_order.limit_price += 1 if order.side.is_bid() else -1
			hidden_order_metadata = dict(ptc_hidden=True, ptc_other_half=visible_order)
			visible_order_metadata = dict(ptc_hidden=False, ptc_other_half=hidden_order)
			self.enter_order(hidden_order, hidden_order_metadata, quiet=True)
			self.enter_order(visible_order, visible_order_metadata, quiet=quiet)
			return
		book = self.bids if order.side.is_bid() else self.asks
		if len(book) == 0:
			book.append(PriceLevel([(order, metadata or {})]))
		elif book[-1].order_has_worse_price(order):
			book.append(PriceLevel([(order, metadata or {})]))
		else:
			for i, price_level in enumerate(book):
				if price_level.order_has_better_price(order):
					book.insert(i, PriceLevel([(order, metadata or {})]))
					break
				elif price_level.order_has_equal_price(order):
					book[i].add_order(order, metadata or {})
					break
		if quiet == False:
			self.history.append(
				dict(
					time=self.owner.current_time,
					type="LIMIT",
					order_id=order.order_id,
					agent_id=order.agent_id,
					side=order.side.value,
					quantity=order.quantity,
					price=order.limit_price,
				)
			)
		if (self.owner.book_logging == True) and (quiet == False):
			self.append_book_log2()
		# Log LOB state after entering order
		bid_data = self.get_l1_bid_data()
		ask_data = self.get_l1_ask_data()
		logger.debug(
			f"LOB after entering order: symbol={self.symbol}, "
			f"best_bid={bid_data[0] if bid_data else None}, "
			f"best_ask={ask_data[0] if ask_data else None}, "
			f"bid_size={bid_data[1] if bid_data else None}, "
			f"ask_size={ask_data[1] if ask_data else None}"
		)

	def cancel_order(
		self,
		order: LimitOrder,
		tag: str = None,
		cancellation_metadata: Optional[Dict] = None,
		quiet: bool = False,
	) -> bool:
		book = self.bids if order.side.is_bid() else self.asks
		if not book:
			return False
		for i, price_level in enumerate(book):
			if not price_level.order_has_equal_price(order):
				continue
			cancelled_order_result = price_level.remove_order(order.order_id)
			if cancelled_order_result is not None:
				cancelled_order, metadata = cancelled_order_result
				if price_level.is_empty:
					del book[i]
				logger.debug("CANCELLED: order %s", order)
				logger.debug(
					"SENT: notifications of order cancellation to agent %s for order %s",
					cancelled_order.agent_id,
					cancelled_order.order_id
				)
				if cancelled_order.is_price_to_comply:
					self.cancel_order(metadata["ptc_other_half"], quiet=True)
				if not quiet:
					self.history.append(
						dict(
							time=self.owner.current_time,
							type="CANCEL",
							order_id=cancelled_order.order_id,
							tag=tag,
							metadata=cancellation_metadata if tag == "auctionFill" else None,
						)
					)
					self.owner.send_message(order.agent_id, OrderCancelledMsg(cancelled_order))
				self.last_update_ts = self.owner.current_time
				if (self.owner.book_logging == True) and (quiet == False):
					self.append_book_log2()
				# Log LOB state after cancellation
				bid_data = self.get_l1_bid_data()
				ask_data = self.get_l1_ask_data()
				logger.debug(
					f"LOB after order cancellation: symbol={self.symbol}, "
					f"best_bid={bid_data[0] if bid_data else None}, "
					f"best_ask={ask_data[0] if ask_data else None}, "
					f"bid_size={bid_data[1] if bid_data else None}, "
					f"ask_size={ask_data[1] if ask_data else None}"
				)
				return True
		return False

	def modify_order(self, order: LimitOrder, new_order: LimitOrder) -> None:
		if order.order_id != new_order.order_id:
			return
		book = self.bids if order.side.is_bid() else self.asks
		for price_level in book:
			if not price_level.order_has_equal_price(order):
				continue
			if price_level.update_order_quantity(order.order_id, new_order.quantity):
				self.history.append(
					dict(
						time=self.owner.current_time,
						type="MODIFY",
						order_id=order.order_id,
						new_side=order.side.value,
						new_quantity=new_order.quantity,
					)
				)
				logger.debug("MODIFIED: order %s", order)
				logger.debug(
					"SENT: notifications of order modification to agent %s for order %s",
					new_order.agent_id,
					new_order.order_id
				)
				self.owner.send_message(order.agent_id, OrderModifiedMsg(new_order))
				self.last_update_ts = self.owner.current_time
				if self.owner.book_logging == True:
					self.append_book_log2()
				# Log LOB state after modification
				bid_data = self.get_l1_bid_data()
				ask_data = self.get_l1_ask_data()
				logger.debug(
					f"LOB after order modification: symbol={self.symbol}, "
					f"best_bid={bid_data[0] if bid_data else None}, "
					f"best_ask={ask_data[0] if ask_data else None}, "
					f"bid_size={bid_data[1] if bid_data else None}, "
					f"ask_size={ask_data[1] if ask_data else None}"
				)

	def partial_cancel_order(
		self,
		order: LimitOrder,
		quantity: int,
		tag: str = None,
		cancellation_metadata: Optional[Dict] = None,
	) -> None:
		if order.order_id == 19653081:
			print("inside OB partialCancel")
		book = self.bids if order.side.is_bid() else self.asks
		new_order = deepcopy(order)
		new_order.quantity -= quantity
		for price_level in book:
			if not price_level.order_has_equal_price(order):
				continue
			if price_level.update_order_quantity(order.order_id, new_order.quantity):
				self.history.append(
					dict(
						time=self.owner.current_time,
						type="CANCEL_PARTIAL",
						order_id=order.order_id,
						quantity=quantity,
						tag=tag,
						metadata=cancellation_metadata if tag == "auctionFill" else None,
					)
				)
				logger.debug("CANCEL_PARTIAL: order %s", order)
				logger.debug(
					"SENT: notifications of order partial cancellation to agent %s for order %s",
					new_order.agent_id,
					quantity
				)
				self.owner.send_message(order.agent_id, OrderPartialCancelledMsg(new_order))
				self.last_update_ts = self.owner.current_time
				if self.owner.book_logging == True:
					self.append_book_log2()
				# Log LOB state after partial cancellation
				bid_data = self.get_l1_bid_data()
				ask_data = self.get_l1_ask_data()
				logger.debug(
					f"LOB after partial cancellation: symbol={self.symbol}, "
					f"best_bid={bid_data[0] if bid_data else None}, "
					f"best_ask={ask_data[0] if ask_data else None}, "
					f"bid_size={bid_data[1] if bid_data else None}, "
					f"ask_size={ask_data[1] if ask_data else None}"
				)

	def replace_order(
		self,
		agent_id: int,
		old_order: LimitOrder,
		new_order: LimitOrder,
	) -> None:
		if self.cancel_order(old_order, quiet=True) == True:
			self.history.append(
				dict(
					time=self.owner.current_time,
					type="REPLACE",
					old_order_id=old_order.order_id,
					new_order_id=new_order.order_id,
					quantity=new_order.quantity,
					price=new_order.limit_price,
				)
			)
			self.handle_limit_order(new_order, quiet=True)
			logger.debug(
				"SENT: notifications of order replacement to agent %s for old order %s, new order %s",
				agent_id,
				old_order.order_id,
				new_order.order_id
			)
			self.owner.send_message(agent_id, OrderReplacedMsg(old_order, new_order))
		if self.owner.book_logging == True:
			self.append_book_log2()
		# Log LOB state after replacement
		bid_data = self.get_l1_bid_data()
		ask_data = self.get_l1_ask_data()
		logger.debug(
			f"LOB after order replacement: symbol={self.symbol}, "
			f"best_bid={bid_data[0] if bid_data else None}, "
			f"best_ask={ask_data[0] if ask_data else None}, "
			f"bid_size={bid_data[1] if bid_data else None}, "
			f"ask_size={ask_data[1] if ask_data else None}"
		)

	def append_book_log2(self):
		row = {
			"QuoteTime": self.owner.current_time,
			"bids": np.array(self.get_l2_bid_data(depth=self.owner.book_log_depth)),
			"asks": np.array(self.get_l2_ask_data(depth=self.owner.book_log_depth)),
		}
		self.book_log2.append(row)
		# Log book_log2 entry for debugging
		logger.debug(
			f"Appended to book_log2: symbol={self.symbol}, "
			f"bids={row['bids'].tolist()}, asks={row['asks'].tolist()}"
		)

	def get_l1_bid_data(self) -> Optional[Tuple[int, int]]:
		if len(self.bids) == 0:
			return None
		index = 0
		while index < len(self.bids) and not self.bids[index].total_quantity > 0:
			index += 1
		if index >= len(self.bids):
			return None
		return self.bids[index].price, self.bids[index].total_quantity

	def get_l1_ask_data(self) -> Optional[Tuple[int, int]]:
		if len(self.asks) == 0:
			return None
		index = 0
		while index < len(self.asks) and not self.asks[index].total_quantity > 0:
			index += 1
		if index >= len(self.asks):
			return None
		return self.asks[index].price, self.asks[index].total_quantity

	def get_l2_bid_data(self, depth: int = sys.maxsize) -> List[Tuple[int, int]]:
		return list(
			filter(
				lambda x: x[1] > 0,
				[(price_level.price, price_level.total_quantity) for price_level in self.bids[:depth]],
			)
		)

	def get_l2_ask_data(self, depth: int = sys.maxsize) -> List[Tuple[int, int]]:
		return list(
			filter(
				lambda x: x[1] > 0,
				[(price_level.price, price_level.total_quantity) for price_level in self.asks[:depth]],
			)
		)

	def get_l3_bid_data(self, depth: int = sys.maxsize) -> List[Tuple[int, List[int]]]:
		return [
			(price_level.price, [order.quantity for order, _ in price_level.visible_orders])
			for price_level in self.bids[:depth]
		]

	def get_l3_ask_data(self, depth: int = sys.maxsize) -> List[Tuple[int, List[int]]]:
		return [
			(price_level.price, [order.quantity for order, _ in price_level.visible_orders])
			for price_level in self.asks[:depth]
		]

	def get_transacted_volume(self, lookback_period: str = "10min") -> Tuple[int, int]:
		window_start = self.owner.current_time - str_to_ns(lookback_period)
		buy_transacted_volume = 0
		sell_transacted_volume = 0
		for time, volume in reversed(self.buy_transactions):
			if time < window_start:
				break
			buy_transacted_volume += volume
		for time, volume in reversed(self.sell_transactions):
			if time < window_start:
				break
			sell_transacted_volume += volume
		return (buy_transacted_volume, sell_transacted_volume)

	def get_imbalance(self) -> Tuple[float, Optional[Side]]:
		bid_vol = sum(price_level.total_quantity for price_level in self.bids)
		ask_vol = sum(price_level.total_quantity for price_level in self.asks)
		if bid_vol == ask_vol:
			return (0, None)
		elif bid_vol == 0:
			return (1.0, Side.ASK)
		elif ask_vol == 0:
			return (1.0, Side.BID)
		elif bid_vol < ask_vol:
			return (1 - bid_vol / ask_vol, Side.ASK)
		else:
			return (1 - ask_vol / bid_vol, Side.BID)

	def get_L1_snapshots(self):
		best_bids = []
		best_asks = []
		def safe_first(x):
			return x[0] if len(x) > 0 else np.array([None, None])
		for d in self.book_log2:
			best_bids.append([d["QuoteTime"]] + safe_first(d["bids"]).tolist())
			best_asks.append([d["QuoteTime"]] + safe_first(d["asks"]).tolist())
		best_bids = np.array(best_bids)
		best_asks = np.array(best_asks)
		return {"best_bids": best_bids, "best_asks": best_asks}

	def bids_padding(self, book, nlevels):
		n = book.shape[0]
		if n == 0:
			return np.zeros((nlevels, 2), dtype=int)
		if n >= nlevels:
			return book[:nlevels, :]
		else:
			lowestprice = book[-1, 0] if len(book.shape) == 2 else book[0]
			npad = nlevels - n
			pad = np.transpose(
				np.array(
					[-1 + np.arange(lowestprice, lowestprice - npad, -1, dtype=int), np.zeros(npad, dtype=int)]
				)
			)
			if len(pad.shape) == 1:
				pad = pad.reshape(1, 2)
			return np.concatenate([book, pad])

	def asks_padding(self, book, nlevels):
		n = book.shape[0]
		if n == 0:
			return np.zeros((nlevels, 2), dtype=int)
		if n >= nlevels:
			return book[:nlevels, :]
		else:
			highestprice = book[-1, 0] if len(book.shape) == 2 else book[0]
			npad = nlevels - n
			pad = np.transpose(
				np.array(
					[1 + np.arange(highestprice, highestprice + npad, 1, dtype=int), np.zeros(npad, dtype=int)]
				)
			)
			if len(pad.shape) == 1:
				pad = pad.reshape(1, 2)
			return np.concatenate([book, pad])

	def get_L2_snapshots(self, nlevels):
		times, bids, asks = [], [], []
		for x in self.book_log2:
			times.append(x["QuoteTime"])
			bids.append(self.bids_padding(x["bids"], nlevels))
			asks.append(self.asks_padding(x["asks"], nlevels))
		bids = np.array(bids)
		asks = np.array(asks)
		times = np.array(times)
		return {"times": times, "bids": bids, "asks": asks}

	def get_l3_itch(self):
		history_l3 = pd.DataFrame(self.history)
		history_l3.loc[history_l3.tag == "auctionFill", "type"] = "EXEC"
		history_l3.loc[history_l3.tag == "auctionFill", "quantity"] = history_l3.loc[
			history_l3.tag == "auctionFill", "metadata"
		].apply(lambda x: x["quantity"])
		history_l3.loc[history_l3.tag == "auctionFill", "price"] = history_l3.loc[
			history_l3.tag == "auctionFill", "metadata"
		].apply(lambda x: x["price"])
		history_l3["printable"] = np.nan
		history_l3["stock"] = np.nan
		if not "REPLACE" in history_l3.type.unique():
			history_l3["new_order_id"] = np.nan
			history_l3["old_order_id"] = np.nan
		history_l3.loc[history_l3.type == "REPLACE", "order_id"] = history_l3.loc[
			history_l3.type == "REPLACE", "old_order_id"
		]
		history_l3.loc[history_l3.type == "EXEC", "side"] = np.nan
		history_l3["type"] = history_l3["type"].replace(
			{"LIMIT": "ADD", "CANCEL_PARTIAL": "CANCEL", "CANCEL": "DELETE", "EXEC": "EXECUTE"}
		)
		history_l3["side"] = history_l3["side"].replace({"ASK": "S", "BID": "B"})
		history_l3["time"] = history_l3["time"] - ns_date(history_l3["time"])
		history_l3["price"] = history_l3["price"] * 100
		history_l3 = history_l3[
			[
				"time",
				"stock",
				"type",
				"order_id",
				"side",
				"quantity",
				"price",
				"new_order_id",
				"printable",
			]
		]
		history_l3 = history_l3.rename(
			columns={"time": "timestamp", "order_id": "reference", "new_order_id": "new_reference", "quantity": "shares"}
		)
		return history_l3

	def pretty_print(self, silent: bool = True) -> Optional[str]:
		assert self.last_trade is not None
		book = "{} order book as of {}\n".format(self.symbol, self.owner.current_time)
		book += "Last trades: simulated {:d}, historical {:d}\n".format(
			self.last_trade,
			self.owner.oracle.observe_price(
				self.symbol, self.owner.current_time, sigma_n=0, random_state=self.owner.random_state
			),
		)
		book += "{:10s}{:10s}{:10s}\n".format("BID", "PRICE", "ASK")
		book += "{:10s}{:10s}{:10s}\n".format("---", "-----", "---")
		for quote, volume in self.get_l2_ask_data()[-1::-1]:
			book += "{:10s}{:10s}{:10s}\n".format("", "{:d}".format(quote), "{:d}".format(volume))
		for quote, volume in self.get_l2_bid_data():
			book += "{:10s}{:10s}{:10s}\n".format("{:d}".format(volume), "{:d}".format(quote), "")
		if silent:
			return book
		else:
			print(book)
			return None
