from multiprocessing import Pool

from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from operators.initialisation import initialisation
from stats.stats import get_stats, stats
from utilities.algorithm.initialise_run import pool_init
from utilities.stats import trackers
import numpy as np


def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    get_stats(individuals)

    #Code to generate gaussian distribution for mutation rates
    x = np.linspace(0.5,0.9,params['GENERATIONS'])
    def normal_dist(x , mean , sd):
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density
 
    #Calculate mean and Standard deviation.
    mean = np.mean(x)
    sd = np.std(x)
 
    #Apply function to the data.
    gaussian_mutation = normal_dist(x,mean,sd)


    # Traditional GE
    for generation in range(1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        # b = [0.16344453, 0.34601207, 0.44428829, 0.34601207, 0.16344453]
        # b1 = [0.16344453, 0.34601207, 0.44428829, 0.34601207, 0.16344453]
        # # c = b1[0]
        # b1.pop(0)
        if generation == 1:
            params['MUTATION_PROBABILITY'] = 0.5
            params['CROSSOVER_PROBABILITY'] = 0.1
            # print(gaussian_mutation[0])
        elif generation == 2:
            params['MUTATION_PROBABILITY'] = 0.6
            params['CROSSOVER_PROBABILITY'] = 0.1
            # print(gaussian_mutation[1])
        elif generation == 3:
            params['MUTATION_PROBABILITY'] = 0.7
            params['CROSSOVER_PROBABILITY'] = 0.1
            # print(gaussian_mutation[2])
        elif generation == 4:
            params['MUTATION_PROBABILITY'] = 0.8
            # print(gaussian_mutation[3])
            params['CROSSOVER_PROBABILITY'] = 0.1
        elif generation == 5:
            params['MUTATION_PROBABILITY'] = 0.01
            # print(gaussian_mutation[4])
            params['CROSSOVER_PROBABILITY'] = 0.9
        elif generation == 6:
            params['MUTATION_PROBABILITY'] = 0.01
            # print(gaussian_mutation[4])
            params['CROSSOVER_PROBABILITY'] = 0.9
        elif generation == 7:
            params['MUTATION_PROBABILITY'] = 0.01
            # print(gaussian_mutation[4])
            params['CROSSOVER_PROBABILITY'] = 0.9
        elif generation == 8:
            params['MUTATION_PROBABILITY'] = 0.01
            # print(gaussian_mutation[4])
            params['CROSSOVER_PROBABILITY'] = 0.9
        else:
            pass






        

        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    individuals = trackers.state_individuals

    if params['MULTICORE']:
        # initialize pool once, if multi-core is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        b = [0.16344453, 0.34601207, 0.44428829, 0.34601207, 0.16344453]
        b1 = [0.16344453, 0.34601207, 0.44428829, 0.34601207, 0.16344453]
        # c = b1[0]
        # b1.pop(0)
        # params['MUTATION_PROBABILITY'] = c
        if generation == 1:
            params['MUTATION_PROBABILITY'] = b1[0]
            print(b1[0])
        if generation == 2:
            params['MUTATION_PROBABILITY'] = b1[1]
            print(b1[1])



        # New generation
        individuals = params['STEP'](individuals)

    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals
