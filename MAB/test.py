from numpy import random
from stand import Stand
from agent import Agent


stand = Stand(0)
agent = Agent(stand)

agent.run(100)
