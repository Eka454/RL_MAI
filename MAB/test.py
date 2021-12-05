from numpy import random
from stand import Stand
from stats import Stats
from agent import Agent


ARM_COUNT = 10
STEPS = 100
EXPS = 10
Q_INIT = 0

rewards = random.uniform(10, 17, ARM_COUNT)

stand = Stand(Q_INIT, rewards)
stats = Stats(stand.arms, EXPS, STEPS)
agent = Agent(stand, stats)

agent.run(EXPS, STEPS)
