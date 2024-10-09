from scipy.optimize import minimize
import numpy as np

demand = np.array([5, 10, 10, 7, 3, 7, 1, 0, 0, 0, 8])
orders = np.array([0.] * len(demand))

def objective(orders):
    return np.sum(orders)


def items_in_stock(orders):
    """In-equality Constraint: Idea is to keep the balance of stock and demand.
    Cumulated stock should be greater than demand. Also, demand should never cross the stock.
    """
    stock = 0
    stock_penalty = 0
    for i in range(len(orders)):
        stock += orders[i]
        stock -= demand[i]
        if stock < 0:
            stock_penalty -= abs(stock)
    return stock_penalty


def four_weeks_order_distance(orders):
    """Equality Constraint: An order can't be placed until four weeks after any other order.
    """
    violation_count = 0
    for i in range(len(orders) - 6):
        if orders[i] != 0.:
            num_orders = orders[i + 1: i + 5].sum()
            violation_count -= num_orders
    return violation_count


def four_weeks_from_end(orders):
    """Equality Constraint: No orders in the last 4 weeks
    """
    return orders[-4:].sum()


con1 = {'type': 'ineq', 'fun': items_in_stock} # Forces value to be greater than zero.
con2 = {'type': 'eq', 'fun': four_weeks_order_distance} # Forces value to be zero.
con3 = {'type': 'eq', 'fun': four_weeks_from_end} # Forces value to be zero.
cons = [con1, con2, con3]

b = [(0, 100)]
bnds = b * len(orders)

x0 = orders
x0[0] = 10.

res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
               options={'eps': 1})
print(res)