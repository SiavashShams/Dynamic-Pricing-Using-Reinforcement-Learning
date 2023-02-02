def train(policy, env, num_episodes):
    rewards = []
    steps_per_episode = []
    tickets_sold_epis = []
    demand_lvl_epis = []
    prices_epis = []
    for episode in range(num_episodes):
        step = 0
        state = env.reset()
        total_reward = 0
        demand_lvl = []
        tickets_sold = []
        prices = []
        while not env.done:
            step += 1
            action = policy.select_action(state)
            price = action
            state, reward, env.done, demand = env.step(price)
            policy.update(state, action, reward)
            total_reward += reward
            demand_lvl.append(demand)
            tickets_sold.append(50 - env.tickets_left)
            prices.append(price)
        rewards.append(total_reward)
        steps_per_episode.append(step)
        tickets_sold_epis.append(tickets_sold)
        demand_lvl_epis.append(demand_lvl)
        prices_epis.append(prices)

    return rewards, steps_per_episode, tickets_sold_epis, demand_lvl_epis, prices_epis
