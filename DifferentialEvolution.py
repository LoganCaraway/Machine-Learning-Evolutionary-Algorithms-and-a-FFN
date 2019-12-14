import random
import copy
import MathAndStats as ms

class DifferentialEvolution:

    def __init__(self, prob_target, beta, population_size, max_generations):
        self.prob_target = prob_target
        self.beta = beta
        self.population_size = population_size
        self.max_generations = max_generations
        self.x = "best"
        self.y = 1
        self.z = "bin"

    def optimize(self, algorithm, parameter_struct, training_set):
        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        population = self.initializePopulation(algorithm, parameter_struct)
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[:validation_index], population)
        for generation in range(1, self.max_generations):
            print("\n\nDE: Working on generation", generation, "of", self.max_generations, "for", len(parameter_struct) - 1,
                  "layer MLP")
            print("Using: probability of using target gene=", self.prob_target, "beta=", self.beta,
                  "population size=", self.population_size)
            next_generation = []
            next_generation_fitness = []
            for target_vector_index in range(self.population_size):
                target_vector = population[target_vector_index]
                target_fitness = population_fitness[target_vector_index]
                trial_vector = self.mutate(population, population_fitness)
                offspring = self.crossover(target_vector, trial_vector)
                offspring_fitness = \
                    self.evaluateGroupFitness(algorithm, parameter_struct, training_set[:validation_index], [trial_vector])[0]
                if offspring_fitness >= target_fitness:
                    next_generation.append(offspring)
                    next_generation_fitness.append(offspring_fitness)
                    print("Fitness from offspring: ", offspring_fitness)
                else:
                    next_generation.append(target_vector)
                    next_generation_fitness.append(target_fitness)
                    print("Fitness from target: ", target_fitness)
            population = next_generation
            population_fitness = next_generation_fitness
            average_fitness = ms.getMean(population_fitness, len(population_fitness))
            print("\n\nDE: Average fitness for the generation: ", average_fitness)
            best_fitness_index = 0
            for index in range(1, len(population)):
                if population_fitness[index] > population_fitness[best_fitness_index]:
                    best_fitness_index = index
            print("Best fitness for generation:", population_fitness[best_fitness_index])
        # evaluate the final population using the validation set to help fight overfitting
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[validation_index:], population)
        # return most fit chromosome
        best_fitness_index = 0
        for index in range(1, len(population)):
            if population_fitness[index] > population_fitness[best_fitness_index]:
                best_fitness_index = index
        most_fit_chromosome = population[best_fitness_index]
        return most_fit_chromosome


    # initialize a population of vectors with random weights along a uniform distribution
    def initializePopulation(self, algorithm, parameter_struct):
        # find length of chromosome
        length, min, max = algorithm.getParameters(parameter_struct)

        # initialize population chromosomes
        population = []
        # generate chromosomes in the population
        for vector in range(self.population_size):
            vect = []
            for weight_index in range(length):
                vect.append(random.uniform(min, max))
            population.append(vect)
        return population


    # evaluate the fitness of a group of vectors
    def evaluateGroupFitness(self, algorithm, parameter_struct, testing_set, group):
        group_fitness = []
        for vector_num in range(len(group)):
            group_fitness.append(algorithm.getFitness(testing_set, parameter_struct, group[vector_num]))
        return group_fitness

    # implements mutation with y difference vectors
    def mutate(self, population, population_fitness):
        if self.x == "rand":
            trial_vector = copy.deepcopy(population[random.randint(0, self.population_size - 1)])
        # default: self.x == "best"
        else:
            best_fitness_index = 0
            for index in range(1, self.population_size):
                if population_fitness[index] > population_fitness[best_fitness_index]:
                    best_fitness_index = index
            trial_vector = copy.deepcopy(population[best_fitness_index])
        for gene in range(len(trial_vector)):
            difference = 0
            for diff in range(self.y):
                difference += self.beta * (population[random.randint(0, self.population_size - 1)][gene] -
                                 population[random.randint(0, self.population_size - 1)][gene])
            trial_vector[gene] += difference
        return trial_vector

    # implements uniform crossover
    def crossover(self, target_vector, trial_vector):
        new_vector = []
        for gene in range(len(target_vector)):
            if random.uniform(0, 1) <= self.prob_target:
                new_vector.append(target_vector[gene])
            else:
                new_vector.append(trial_vector[gene])
        return new_vector