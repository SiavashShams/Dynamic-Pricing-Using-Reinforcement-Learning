from RL_dynamic_pricing.policy import PricingPolicy
from RL_dynamic_pricing.environment import TicketSalesEnv
from RL_dynamic_pricing.training import train
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = TicketSalesEnv(30, 50, 100, 200)
    policy = PricingPolicy(env)
    num_episodes = 200
    rewards, steps, tickets_sold_epis, demand_lvl_epis, prices_epis = train(policy, env, num_episodes)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Revenue')
    plt.show()
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()

    for i in (10, 150):
        plt.plot(tickets_sold_epis[i])
        plt.title("Iteration %i" % (i))
        plt.xlabel('Day')
        plt.ylabel('Tickets Sold')
        plt.show()
        plt.plot(demand_lvl_epis[i], "red")
        plt.title("Iteration %i" % (i))
        plt.xlabel('Day')
        plt.ylabel('Demand Level')
        plt.show()
        plt.plot(prices_epis[i], "green")
        plt.title("Iteration %i" % (i))
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.show()
