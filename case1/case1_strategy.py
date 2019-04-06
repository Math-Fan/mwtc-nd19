import argparse
import random
import numpy as np
import protos
from sympy import *

from client.exchange_service.client import BaseExchangeServerClient
from protos.order_book_pb2 import Order
from protos.service_pb2 import PlaceOrderResponse
from protos.exchange_pb2 import MarketUpdate

current_position = np.zeros([6,1])

def prc_deviation_calc(prc_vec, expected_prc_vec = np.array([100.4478, 100.9738, 101.1336, 101.1543, 101.2378, 101.6016])):
        # Pick out mid_market_price vector
        prc_vec_mid = prc_vec[:,2]
        
        
        # Generate 6x6 matrix of expected price differences among assets
        exp_prc_diffs = np.zeros([6,6])
        for iexp1 in np.arange(len(expected_prc_vec)):
            for iexp2 in np.arange(len(expected_prc_vec)):
                exp_prc_diffs[iexp1,iexp2] = expected_prc_vec[iexp2] - expected_prc_vec[iexp1]
        
        # Generate 6x6 matrix of actual price differences among assets
        act_prc_diffs = np.zeros([6,6])
        for iact1 in np.arange(len(prc_vec_mid)):
            for iact2 in np.arange(len(prc_vec_mid)):
                act_prc_diffs[iact1,iact2] = prc_vec_mid[iact2] - prc_vec_mid[iact1]
                
        # Calculate difference between elements in actual price differences and expected price differences
        mag_diff_arr = act_prc_diffs - exp_prc_diffs
        return mag_diff_arr
        

def alloc_calc(prc_abnormality_arr):
    # Serialize price abnormalities into column vector (first extract upper triangle of data)
    prc_abnormality_arr = np.triu(prc_abnormality_arr)
    prc_abnormality_col = np.concatenate(prc_abnormality_arr)
    
    # Remove the zero entries from this column vector
    prc_abnormality_col = prc_abnormality_col[np.nonzero(prc_abnormality_col)]
    # Generate coefficient matrix
    A = np.zeros([16,6])
    Arow = 0
    while(Arow<14):
        for ia1 in np.arange(6):
            for ia2 in np.arange(ia1+1,6):
                A[Arow,ia1] = 1
                A[Arow,ia2] = 1
                Arow += 1
    # Append the quantity restriction terms
    A[15,:]= np.array([1, 1, 1, 1, 1, 1])
    prc_abnormality_col = np.append(prc_abnormality_col,0)
    
    # Transpose coefficient matrix for least squares approximation
    A_T = np.transpose(A)
    
    # Find both sides of least squares equation, and form matrix to row reduce
    left_side_matrix = np.dot(A_T, A)
    right_side_matrix = np.dot(A_T, prc_abnormality_col)
    total_matrix = np.c_[left_side_matrix,right_side_matrix]
    
    total_matrix_rref = Matrix(total_matrix).rref()
    return np.array([total_matrix_rref[0].col(-1)])
    
    

class ExampleMarketMaker(BaseExchangeServerClient):
    """A simple market making bot - shows the basics of subscribing
    to market updates and sending orders"""

    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)

        self._orderids = set([])
    
    def _make_order(self, asset_code, quantity, base_price, spread):
        return Order(asset_code = asset_code, quantity=quantity,
                     order_type = Order.ORDER_LMT,
                     price = base_price-spread/2 if quantity>0 else base_price+spread/2,
                     competitor_identifier = self._comp_id)
                     
    def _cancel_order(self, order_id):
        return CancelOrderRequest(order_id = order_id) 
    
    def handle_exchange_update(self, exchange_update_response):
        global current_position
        
        """# method for cancelling existing orders
        cancel_resp = self.cancel_order(self._cancel_order(order_id))
        if type(cancel_resp) != CancelOrderResponse:
            print(cancel_resp)
        else:
            self.orderids.pop(cancel_resp.order_id)
        """
        
        # Get current price data
        current_prices = np.zeros([6,3])
        for i,update in enumerate(exchange_update_response.market_updates):
            current_prices[i,:] = np.array([update.bids[0].price, update.asks[0].price, update.mid_market_price])
        
        # Implement pricing strategy    
        prc_abnormality_arr = prc_deviation_calc(current_prices)
        weight_vec = 20*alloc_calc(prc_abnormality_arr)
        weight_int = weight_vec.astype(int)
        
        # Check for filled quantities
        filled_order_quantity_vec = np.zeros([6,2])
        filled_order_asset_vec = np.chararray([6,1])
        for i, update in enumerate(exchange_update_response.fills):
            filled_order_quantity_vec[i,:] = np.array([update.order.quantity,update.order.remaining_quantity])
            filled_order_asset_vec[i,0] = update.order.asset_code
        print("Fills:")
        print(filled_order_asset_vec)
        print(filled_order_quantity_vec)
        
        # Calculate current position
        current_position[:,0] = current_position[:,0] + filled_order_quantity_vec[:,0] - filled_order_quantity_vec[:,1]
        print("Current Position:\n",current_position)
        
        # Ordering Logic
        for i, asset_code in enumerate(["K", "M", "N", "Q", "U", "V"]):
            quantity = weight_int[0,i] - current_position[i,0]
            quantity = quantity.astype(int)
            base_price = round(current_prices[i,2],2)
            spread = 4
        
            order_resp = self.place_order(self._make_order(asset_code, quantity,
                base_price, spread))
            
            # implement error checking
            if type(order_resp) != PlaceOrderResponse:
                print(order_resp)
            else:
                self._orderids.add(order_resp.order_id)
                
        # Track PnL
        print("PnL:",exchange_update_response.competitor_metadata.pnl)
        print("\n\n")


           
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
