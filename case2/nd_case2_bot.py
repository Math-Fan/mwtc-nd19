import argparse
import random

from src.client.exchange_service.client import BaseExchangeServerClient
from src.protos.order_book_pb2 import Order
from src.protos.service_pb2 import PlaceOrderResponse
import py_vollib.black-scholes as bs
from datetime import timedelta, time
import py_vollib.black-scholes.greeks.analytical as bsga
import py_vollib.black-scholes.greeks.numerical as bsgn
import multiprocessing as mp
import math


class NDMarketMaker(BaseExchangeServerClient):

    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)

        self.START_TIME = datetime.datetime.now()
        self._orderids = set([])
        self.asset_codes = { "C98PHX":98, "P98PHX":98,
                             "C99PHX":99, "P99PHX":99,
                             "C100PHX":100, "P100PHX":100,
                             "C101PHX":101, "P101PHX":101,
                             "C102PHX":102, "P102PHX":102, }
        self.underlying_price


    def _make_order(self, asset_code, quantity, base_price, spread, bid=True):
        return Order(asset_code = asset_code, quantity=quantity if bid else -1*quantity,
                     order_type = Order.ORDER_LMT,
                     price = base_price-spread/2 if bid else base_price+spread/2,
                     competitor_identifier = self._comp_id)


    def _get_time_to_exp():
        return (3-((datetime.datetime.now()-self.START_TIME).total_seconds()*1./900.))*(1./12.)

    
    def get_spread(volatility):
        return volatility*(get_time_to_exp()) + math.log(2)


    def weighted_price(mid_price, imbalance, best_ask, best_bid):
        return imbalance*best_ask + (1-imbalance)*best_bid
        

    def get_measured_price(mid_price, imbalance, best_ask, best_bid, **kwargs):
        method = kwargs.get(method, "weighted")
        if method == "weighted":
            return weighted_price(mid_price, imbalance, best_ask, best_bid)


    MIN_SPREAD = 0.01
    def generate_limit_order(asset_code, measured_price, volatility, best_ask, best_bid, **kwargs):
        min_spread = kwargs("min_spread", MIN_SPREAD)
        bs_price = bs.black_scholes(asset_code[0].lower(), self.underlying_price, _
                                    asset_codes[asset_code], self._get_time_to_exp, 0, _
                                    volatility)
        if bs_price < measured_price:
            ask_resp = self.place_order(self._make_order(asset_code, 30, _
                                                         measured_price, _
                                                         get_spread(volatility), _
                                                         False)
       if type(ask_resp) != PlaceOrderResponse:
            print(ask_resp)
       else:
            self._orderids.add(ask_resp.order_id)


    def handle_exchange_update(self, exchange_update_response):
        for update in exhange_update_response.market_updates:
           generate_limit_order(update.asset.asset_code, 


    def tester():
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=str, default="50052")
    parser.add_argument("--client_id", type=str)
    parser.add_argument("--client_private_key", type=str)
    parser.add_argument("--websocket_port", type=int, default=5678)

    args = parser.parse_args()
    host, port, client_id, client_pk, websocket_port = (args.server_host, args.server_port,
                                        args.client_id, args.client_private_key,
                                        args.websocket_port)

    client = ExampleMarketMaker(host, port, client_id, client_pk, websocket_port)
    client.start_updates()
