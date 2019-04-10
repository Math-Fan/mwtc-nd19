import argparse
import random
import numpy as np
import protos
from sympy import *

from client.exchange_service.client import BaseExchangeServerClient
from protos.order_book_pb2 import Order
from protos.service_pb2 import PlaceOrderResponse, CancelOrderResponse
from protos.exchange_pb2 import MarketUpdate

current_position = {"K": 0, "M": 0, "N": 0, "Q": 0, "U": 0, "V": 0}
quantity_to_order = {"K": 0, "M": 0, "N": 0, "Q": 0, "U": 0, "V": 0}
filled_amounts = {"K": 0, "M": 0, "N": 0, "Q": 0, "U": 0, "V": 0}
violation_risk = {"K": 0, "M": 0, "N": 0, "Q": 0, "U": 0, "V": 0}
net_position_error = int(0)
running_total = 0

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
        

def alloc_calc(prc_abnormality_arr,net_position_error):
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
    prc_abnormality_col = np.append(prc_abnormality_col,net_position_error*-1)
    
    # Transpose coefficient matrix for least squares approximation
    A_T = np.transpose(A)
    
    # Find both sides of least squares equation, and form matrix to row reduce
    left_side_matrix = np.dot(A_T, A)
    right_side_matrix = np.dot(A_T, prc_abnormality_col)
    total_matrix = np.c_[left_side_matrix,right_side_matrix]
    
    total_matrix_rref = Matrix(total_matrix).rref()
    return np.array([total_matrix_rref[0].col(-1)])
    
def calc_vector_distance(vec):
    running_total = 0
    for i in np.arange(len(vec)):
        running_total += vec[i,0]**2
    return np.sqrt(running_total)

"""    
def summarizeFills(assets,filled_quantities):
    summarized_fills = np.zeros([6,1])
    for i in np.arange(len(assets)):
        if(assets[i] == "K"):
            summarized_fills[0,0] += filled_quantities[i]
        elif(assets[i] == "M"):
            summarized_fills[1,0] += filled_quantities[i];
        elif(assets[i] == "N"):
            summarized_fills[2,0] += filled_quantities[i];
        elif(assets[i] == "Q"):
            summarized_fills[3,0] += filled_quantities[i];
        elif(assets[i] == "U"):
            summarized_fills[4,0] += filled_quantities[i];
        elif(assets[i] == "V"):
            summarized_fills[5,0] += filled_quantities[i];
    return summarized_fills
"""            

class ExampleMarketMaker(BaseExchangeServerClient):
    """A simple market making bot - shows the basics of subscribing
    to market updates and sending orders"""

    def __init__(self, *args, **kwargs):
        BaseExchangeServerClient.__init__(self, *args, **kwargs)

        self._orderids = set([])
    
    def _make_order(self, asset_code, quantity, price_input):
        return Order(asset_code = asset_code, quantity=quantity,
                     order_type = Order.ORDER_LMT,
                     price = price_input,
                     competitor_identifier = self._comp_id)
    
    def handle_exchange_update(self, exchange_update_response):
        global current_position, quantity_to_order, filled_amounts, violation_risk, net_position_error, running_total
        
        # Get current price data
        current_prices = np.zeros([6,3])
        for i,update in enumerate(exchange_update_response.market_updates):
            current_prices[i,:] = np.array([update.bids[0].price, update.asks[0].price, update.mid_market_price])
        print("Current Prices:")
        print(current_prices)
        
        
        # Implement pricing strategy    
        prc_abnormality_arr = prc_deviation_calc(current_prices)
        weight_vec = 20*alloc_calc(prc_abnormality_arr,net_position_error)
        weight_int = weight_vec.astype(int)
        print("Error Correction:", net_position_error)
        print("Weight Vector:")
        print(weight_int)
        
        # Check for filled quantities and update current position
        for i, update in enumerate(exchange_update_response.fills):
            print(update.filled_quantity)
            filled_amounts[update.order.asset_code] = update.filled_quantity
            current_position[update.order.asset_code] += update.filled_quantity
        print("Quantity to Order:")
        print(quantity_to_order)
        
        print("Order IDS:")
        print(len(self._orderids))
        
        # Cancel Existing Orders
        current_outstanding_orders = self._orderids.copy()
        for iorder_id in current_outstanding_orders:
            print(iorder_id)
            cancel_resp = self.cancel_order(iorder_id)
            if(cancel_resp.success != True):
                print("Error Canceling Order",cancel_resp)
            else:
                try:
                    self._orderids.remove(iorder_id)
                except:
                    pass
        
        print(len(self._orderids))
        
        # Calculate the ideal quantity to order
        for i, asset_code in enumerate(["K", "M", "N", "Q", "U", "V"]):
            quantity_to_order[asset_code] = weight_int[0,i] - filled_amounts[asset_code]
            violation_risk[asset_code] = current_position[asset_code] + quantity_to_order[asset_code]
        
        
        
        # Ordering Logic
        if(sum(list(violation_risk.values()))<50):
            for i, asset_code in enumerate(["K", "M", "N", "Q", "U", "V"]):
                quantity = quantity_to_order[asset_code]
                running_total += abs(quantity)
                print("Quantity to Order:", quantity)            
                
                # Make Orders
                order_resp = 0
                if(quantity > 0):
                    order_resp = self.place_order(self._make_order(asset_code, quantity, round(current_prices[i,0],2)))
                elif(quantity < 0):
                    order_resp = self.place_order(self._make_order(asset_code, quantity, round(current_prices[i,1],2)))
                
                # implement error checking
                if type(order_resp) != PlaceOrderResponse:
                    print(order_resp)
                else:
                    self._orderids.add(order_resp.order_id)
                
        # Calculate feedback
        print("Error Correction:")
        print(list(current_position.values()))
        print(sum(list(current_position.values())))
        net_position_error = sum(list(current_position.values()))/20
        
        # Track PnL
        print("PnL:",exchange_update_response.competitor_metadata.pnl)
        print("Running Total:",running_total)
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
