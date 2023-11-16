import random
from deap import base, creator, tools, algorithms

# Create a maximization fitness class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the evaluation function for the individual
def eval_func(individual):
    x, y, z = individual
    result = 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)
    return result,


# Create a toolbox for the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    population = toolbox.population(n=50)
    generations = 20
    crossover_prob = 0.7
    mutation_prob = 0.2

# Run the genetic algorithm using the mu + lambda evolutionary strategy
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=200, cxpb=crossover_prob, mutpb=mutation_prob,
                              ngen=generations, stats=None, halloffame=None, verbose=True)

  # Select the best individual from the final population
    best_individual = tools.selBest(population, k=1)[0]
    best_values = eval_func(best_individual)

 # Print the best individual and its fitness
    print("Best individual:", best_individual)
    print("Best fitness:", best_values)


if __name__ == "__main__":
    main()