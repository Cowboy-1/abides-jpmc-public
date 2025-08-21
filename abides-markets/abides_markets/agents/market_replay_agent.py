# market_replay_agent.py
from abides_core import Agent, NanosecondTime
from abides_markets.messages.order import LimitOrderMsg, CancelOrderMsg
from abides_markets.orders import LimitOrder
import logging

logger = logging.getLogger(__name__)

class MarketReplayAgent(Agent):
    def __init__(self, id, messages, symbol, **kwargs):
        super().__init__(id, **kwargs)
        self.messages = messages
        self.symbol = symbol
        self.exchange_id = 0
        self.current_message_idx = 0

    def kernel_start(self, kernel):
        super().kernel_start(kernel)
        for msg in self.messages:
            self.schedule_wakeup(NanosecondTime(msg['timestamp']))

    def wakeup(self, current_time):
        super().wakeup(current_time)
        while (self.current_message_idx < len(self.messages) and
               self.messages[self.current_message_idx]['timestamp'] <= current_time):
            msg = self.messages[self.current_message_idx]
            if msg['symbol'] != self.symbol:
                logger.warning(f"Symbol mismatch: expected {self.symbol}, got {msg['symbol']}")
            else:
                if msg['type'] == 'LIMIT_ORDER':
                    order = LimitOrder(
                        agent_id=self.id,
                        symbol=self.symbol,
                        quantity=msg['quantity'],
                        side='BID' if msg['is_buy'] else 'ASK',
                        limit_price=msg['price'],
                        timestamp=current_time
                    )
                    logger.debug(f"Sending LIMIT_ORDER: {order}")
                    self.send_message(self.exchange_id, LimitOrderMsg(order=order))
                elif msg['type'] == 'CANCEL_ORDER':
                    order = LimitOrder(
                        agent_id=self.id,
                        symbol=self.symbol,
                        quantity=msg['quantity'],
                        side='BID' if msg['is_buy'] else 'ASK',
                        limit_price=msg['price'],
                        timestamp=current_time
                    )
                    logger.debug(f"Sending CANCEL_ORDER: {order}")
                    self.send_message(self.exchange_id, CancelOrderMsg(order=order))
            self.current_message_idx += 1

    def receive_message(self, current_time, msg):
        super().receive_message(current_time, msg)
        # No action needed for received messages
        pass
