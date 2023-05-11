import neat
import os
import pickle
import time
import random
import visualize
import pandas as pd
import numpy as np

start_time = time.time()
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('Neat_checkpoint_first_try/neat-checkpoint-41')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    # Stworzenie standardowego modelu NEAT w celu szkolenia i dodanie obserwacji statystycznej

    winner = p.run(eval_genomes, 1)
    #with open("best_41.pickle", "wb") as f:
    #    pickle.dump(winner, f)

    #visualize.draw_net(config, winner, True)
    #visualize.draw_net(config, winner, True, prune_unused=True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)


def eval_genomes(genomes, config):
    aktywacja = pd.read_csv('train.csv')
    wyniki = aktywacja['Survived']
    aktywacja = aktywacja.drop(columns=['Survived', 'PassengerId', 'Name'])
    aktywacja['Sex'] = np.where(aktywacja["Sex"] == "female", 0, 1)

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        train_ai(genome1, config, aktywacja, wyniki)


def train_ai(genome1, config, aktywacja, wyniki):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    a = net1.activate(aktywacja.iloc[[1]].values.flatten().tolist())
    print(a)
    genome1.fitness += 0



def NEAT_learn():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)


if __name__ == "__main__":
    aktywacja = pd.read_csv('train.csv')
    wyniki = aktywacja['Survived']
    aktywacja = aktywacja.drop(columns=['Survived', 'PassengerId', 'Name'])
    aktywacja['Sex'] =  np.where(aktywacja["Sex"] == "female", 0, 1)
    aktywacja['Age'] = aktywacja['Age'].fillna(0)
    aktywacja['Age'] = aktywacja['Age'].astype('int')
    aktywacja['Fare'] = aktywacja['Fare'].fillna(0)
    aktywacja['Fare'] = aktywacja['Fare'].astype('int')
    aktywacja['Cabin'] = aktywacja['Cabin'].fillna(0)
    aktywacja['Cabin'] = aktywacja['Cabin'].astype('str')
    print(aktywacja.head())
    test = aktywacja.iloc[[0]].values.flatten().tolist()
    for i in range(len(test)):
        print(test[i])
        print(type(test[i]))
    #print(aktywacja)
    NEAT_learn()


