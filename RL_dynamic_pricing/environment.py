import random
import numpy as np
import scipy.stats
from math import floor


# Define the environment, which will be the ticket sales simulation
class TicketSalesEnv:
    def __init__(self, days_left, tickets_left, demand_level_min, demand_level_max, avg_market_price=100,
                 stddev_market_price=10):
        self.days_left = days_left
        self.tickets_left = tickets_left
        self.demand_level_min = demand_level_min
        self.demand_level_max = demand_level_max
        self.avg_market_price = avg_market_price
        self.stddev_market_price = stddev_market_price
        self.done = False

    def cal_demand(self, days_left):

        if days_left <= 10:
            demand_lvl = random.uniform(self.demand_level_min + 100 / (days_left + 1), self.demand_level_max)
        else:
            demand_lvl = random.uniform(self.demand_level_min, self.demand_level_max)
        return demand_lvl

    def cal_avg_market_price(self, demand_lvl):
        avg_market = self.avg_market_price + demand_lvl * 0.1
        return avg_market

    def reset(self):
        self.days_left = 30
        self.tickets_left = 50
        self.done = False
        return self.observe()

    def observe(self):
        demand_level = self.cal_demand(self.days_left)
        return (self.days_left, self.tickets_left, demand_level)

    def tickets_sold(self, price):
        demand_level = self.cal_demand(self.days_left)
        avg_market_price = self.cal_avg_market_price(demand_level)
        # avg_market_price=self.avg_market_price
        quantity_demanded = floor(
            max(0, demand_level * scipy.stats.norm(avg_market_price, self.stddev_market_price).pdf(price)))
        return min(quantity_demanded, self.tickets_left), demand_level

    def step(self, price):
        tickets_sold, demand_level = self.tickets_sold(price)
        revenue = price * tickets_sold
        self.tickets_left -= tickets_sold
        self.days_left -= 1
        if self.days_left == 0 or self.tickets_left == 0:
            self.done = True
        return self.observe(), revenue, self.done, demand_level
