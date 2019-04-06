import argparse
import random

from src.client.exchange_service.client import BaseExchangeServerClient
from src.protos.order_book_pb2 import Order
from src.protos.service_pb2 import PlaceOrderResponse
import py_vollib.black_scholes as bs
from datetime import timedelta, time
import py_vollib.black_scholes.greeks.analytical as bsga
import py_vollib.black_scholes.greeks.numerical as bsgn
import multiprocessing as mp
import math


class NDMarketMaker(BaseExchangeServerClient):

    def __init_asset_codes():
        asset_codes = {}
        codes = ["C98PHX", "P98PHX",
                 "C99PHX", "P99PHX", 
                 "C100PHX", "P100PHX", 
                 "C101PHX", "P101PHX", 
                 "C102PHX", "P102PHX", ]
        for code in codes:
            asset_codes[code]["strike"] = int(code[1:-3])
            asset_codes[code]["vol"] = 8.*abs(math.log(100./asset_codes[code]["strike"])
            asset_codes[code]["price"] = bs.black_scholes(code[0].lower(), 100, asset_code[code]["strike"],
                                                          0.25, 0, asset_code[code]["vol"])
        return asset_codes


    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)

        self.START_TIME = datetime.datetime.now()
        self._orderids = set([])
        self.asset_codes = __init_asset_codes()
        self.underlying_price = 100


    def _make_order(self, asset_code, quantity, base_price, spread, bid=True):
        return Order(asset_code = asset_code, quantity=quantity if bid else -1*quantity,
                     order_type = Order.ORDER_LMT,
                     price = base_price-spread/2 if bid else base_price+spread/2,
                     competitor_identifier = self._comp_id)


    # computes time to expiration
    def _get_time_to_exp():
        return (3-((datetime.datetime.now()-self.START_TIME).total_seconds()*1./900.))*(1./12.)


    # In development
    # TODO: overload with *args and **kwargs
    def get_spread(volatility):
        return volatility*(get_time_to_exp()) + math.log(2)

    
    # gets spread and base_price using primitive adjustment method
    def _get_spread(bsp, wmp, ask, bid):
        x3 = ask - wmp
        x2 = wmp - bsp
        x1 = bsp - bid
        ask = ask - (x2/(bid-ask))*x3
        bid = bid - (x1/(bid-ask))*x1
        return (ask - bid, (ask + bid)/2)



    # computes weighted price
    def weighted_price(mid_price, imbalance, best_ask, best_bid):
        return imbalance*best_ask + (1-imbalance)*best_bid



    # gets meaured price, designed so we can easily adjust it
    def get_measured_price(mid_price, imbalance, best_ask, best_bid, **kwargs):
        method = kwargs.get(method, "weighted")
        if method == "weighted":
            return weighted_price(mid_price, imbalance, best_ask, best_bid)


    # Sends orders to exchange
    def send_order(asset_code, qty, base_price, spread):
        ask_resp = self.place_order(self._make_order(asset_code, qty, base_price, spread,  False))
        bid_resp = self.place_order(self._make_order(asset_code, qty, base_price, spread,  True))

        if type(ask_resp) != PlaceOrderResponse:
            print(ask_resp)
        else:
            self._orderids.add(ask_resp.order_id)
        
        if type(bid_resp) != PlaceOrderResponse:
            print(bid_resp)
        else:
            self._orderids.add(bid_resp.order_id)



    # Generates then sends orders
    MIN_SPREAD = 0.02
    def generate_limit_order(asset_code, measured_price, volatility, best_ask, best_bid, **kwargs):
        min_spread = kwargs("min_spread", MIN_SPREAD)
        bs_price = bs.black_scholes(asset_code[0].lower(), self.underlying_price, asset_codes[asset_code]["strike"], self._get_time_to_exp, 0, volatility)
        spread, base_price = _get_spread(bs_price, measured_price, best_ask, best_bid)

        if spread < min_spread: return

        send_order(asset_code, 30, base_price, spread)
        asset_codes[asset_code]["vol"] = bs.black_scholes.implied_volatility.implied_volatility(measured_price, asset_codes[asset_code]["strike"],
                                                                                                get_time_to_exp(), 0, asset_code[0].lower())


    # TODO: Handle fills/hedge
    def HANDLE_FILLS():
        pass



    def handle_exchange_update(self, exchange_update_response):
        ''' Method for handling exchange updates
            - gathers and exchange update and responds
            - creates a process for each symbol
            --> (not robust, may not work, haven't experimented with
                memory sharing/dict access)
        '''
        processes = []
        for update in exhange_update_response.market_updates:
            code = update.asset.asset_code
            imbalance = update.bids[0].quantity / (update.bids[0].quantity + update.asks[0].quantity)
            measured_price = get_measured_price(update.mid_market_price, imbalance, update.ask, update,bid)
            p = mp.Process(target=generate_limit_order, args=(code, measured_price, self.asset_codes[code]["vol"], update.ask, update.bid, min_spread=0.02))
            p.start()
        for p in processes:
            p.join()



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
