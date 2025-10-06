from game import Game, Car
import torch
import copy
import numpy as np
from model import Net,Trainer
N_AGENT = 40
CrossOverRate = 0.3
Mutatation_Rate = 0.3
Mutation_Strength = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.model = Net()
        self.reward = 0
    def getState(self, game, car):
        state = [
            car.speed,
        ]
        for i in game.raycast(car):
            state.append(i)
        return state
    def getAction(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        return self.model(state0)
def train(Mutation_Rate = 0.1):
    record = 0
    game = Game(N_AGENT)
    agents = [Agent() for _ in range(N_AGENT)]
    torch.serialization.add_safe_globals([Net])
    trainer = Trainer(Mutation_Rate,Mutation_Strength,CrossOverRate)
    done = False
    best = torch.load('best_model.pth')
    agents[0].model.load_state_dict(copy.deepcopy(best))
    while True:
        actions = []
        for i in range(0,N_AGENT):
            state = agents[i].getState(game, game.cars[i])
            action = agents[i].getAction(state)
            actions.append(action)
            agents[i].reward = game.cars[i].reward
        game.__gamestep__(actions)
        if(game.done):
            done = True
        if done:
            game.__reset__()
            if sorted(agents, key=lambda a: a.reward, reverse=True)[0].reward > record:
                best_agent = sorted(agents, key=lambda a: a.reward, reverse=True)[0]
                best = copy.deepcopy(best_agent.model.state_dict())
                record = float(best_agent.reward)
                torch.save(best, 'best_model.pth')
            trainer.mutation_rate = game.mutation_rate
            trainer.crossover_rate = game.crossover_rate
            trainer.mutation_strength = game.mutation_strength
            trainer.Train(copy.deepcopy(best), copy.deepcopy(sorted(agents, key=lambda a: a.reward, reverse=True)[1].model.state_dict()) , [i.model for i in agents])
            agents[0].model.load_state_dict(best)
            agents[0].model.eval()
            done = False


            print(record)
if __name__ == "__main__":
    train(Mutatation_Rate)