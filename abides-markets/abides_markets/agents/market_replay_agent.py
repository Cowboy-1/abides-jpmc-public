# market_replay_agent.py
from abides_core import Agent, NanosecondTime
from abides_markets.messages.order import LimitOrderMsg, CancelOrderMsg
from abides_markets.orders import LimitOrder, Side
import logging

logger = logging.getLogger(__name__)

class MarketReplayAgent(Agent):
    def __init__(self, id, messages, symbol, **kwargs):
        if 'computation_delay' in kwargs:
            kwargs.pop('computation_delay')
        super().__init__(id, **kwargs)
        self.messages = messages
        self.symbol = symbol
        self.exchange_id = 0
        self.current_message_idx = 0

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        print("Replay agent kernel_starting, scheduling wakeups for", len(self.messages), "messages")
        for msg in self.messages:
            wakeup_time = NanosecondTime(msg['timestamp'])
            if wakeup_time > start_time:
                self.set_wakeup(wakeup_time)
        self.set_wakeup(start_time + 1)

    def wakeup(self, current_time):
        super().wakeup(current_time)
        print("Replay wakeup called at", current_time)
        while (self.current_message_idx < len(self.messages) and
               self.messages[self.current_message_idx]['timestamp'] <= current_time):
            msg = self.messages[self.current_message_idx]
            print(f"Sending {msg['type']} at time {current_time}, idx {self.current_message_idx}")
            if msg['symbol'] != self.symbol:
                logger.warning(f"Symbol mismatch: expected {self.symbol}, got {msg['symbol']}")
            else:
                if msg['type'] == 'LIMIT_ORDER':
                    order = LimitOrder(
                        agent_id=self.id,
                        symbol=self.symbol,
                        quantity=msg['quantity'],
                        side=Side.BID if msg['is_buy'] else Side.ASK,
                        limit_price=msg['price'],
                        time_placed=current_time
                    )
                    logger.debug(f"Sending LIMIT_ORDER: {order}")
                    self.send_message(self.exchange_id, LimitOrderMsg(order=order))
                elif msg['type'] == 'CANCEL_ORDER':
                    order = LimitOrder(
                        agent_id=self.id,
                        symbol=self.symbol,
                        quantity=msg['quantity'],
                        side=Side.BID if msg['is_buy'] else Side.ASK,
                        limit_price=msg['price'],
                        time_placed=current_time
                    )
                    logger.debug(f"Sending CANCEL_ORDER: {order}")
                    self.send_message(self.exchange_id, CancelOrderMsg(order=order))
            self.current_message_idx += 1

    def receive_message(self, current_time, sender_id, msg):
        super().receive_message(current_time, sender_id, msg)
        # No action needed for received messages
        pass
