import random
import copy
import MathAndStats as ms

class EvolutionStrategies:

    # mu: population size
    # rho: number of parents for recombination
    # lmbda: number of generated offspring
    # alpha: scaling multiplier for adaptive sigma
    # sigma: standard deviation
    def __init__(self, mu, rho, lmbda, alpha, sigma, max_generations):
        self.mu = mu
        self.rho = rho
        self.lmbda = lmbda
        self.alpha = alpha
        self.sigma = sigma
        self.max_generations = max_generations
        self.probability_of_offspring = 0

    def optimize(self, algorithm, parameter_struct, training_set):
        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        population = self.initializePopulation(algorithm, parameter_struct)
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[:validation_index],
                                                       population)
        for generation in range(1, self.max_generations):
            # adaptive variance using the 1/5th rule
            if (generation > 0) and (generation % 10 == 0):
                # counvert to probability
                self.probability_of_offspring /= (10 * self.mu)
                print("From the last 10 generations, the average probability of selecting an"
                      "offspring is",self.probability_of_offspring)
                if self.probability_of_offspring > 0.2:
                    self.sigma /= self.alpha
                elif self.probability_of_offspring < 0.2:
                    self.sigma *= self.alpha
                self.probability_of_offspring = 0
            print("\n\nEvolution Strategies: Working on generation", generation, "of",
                  self.max_generations, "\nUsing: mu=",self.mu, "rho=",self.rho,"lmbda=",
                  self.lmbda,"sigma=",self.sigma)
            offspring_group = []
            offspring_group_fitness = []
            for offspring_num in range(self.lmbda):
                if self.rho == 1:
                    offspring = copy.deepcopy(population[random.randint(0, self.mu - 1)])
                elif self.mu == self.rho:
                    offspring = self.recombine(population)
                else:
                    parents = []
                    for parent_num in range(self.rho):
                        parents.append(population[random.randint(0, self.mu - 1)])
                    offspring = self.recombine(parents)
                self.mutate(offspring)
                offspring_fitness = self.evaluateGroupFitness(algorithm, parameter_struct,
                                                              training_set[:validation_index], [offspring])[0]
                offspring_group.append(offspring)
                offspring_group_fitness.append(offspring_fitness)
            population, population_fitness = self.selection(population, offspring_group,
                                                            population_fitness, offspring_group_fitness)
            best_fitness_index = 0
            average_fitness = population_fitness[0]
            for index in range(1, len(population)):
                average_fitness += population_fitness[index]
                if population_fitness[index] > population_fitness[best_fitness_index]:
                    best_fitness_index = index
            average_fitness /= len(population)
            print("Best fitness for generation:", population_fitness[best_fitness_index])
            print("Average fitness for generation:", average_fitness)
        # evaluate the final population using the validation set to help fight overfitting
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct,
                                                       training_set[validation_index:], population)
        # return most fit chromosome
        best_fitness_index = 0
        for index in range(1, len(population)):
            if population_fitness[index] > population_fitness[best_fitness_index]:
                best_fitness_index = index
        most_fit_chromosome = population[best_fitness_index]
        return most_fit_chromosome



    # initialize a population of chromosomes with random weights in [min, max]
    # along a uniform distribution
    def initializePopulation(self, algorithm, parameter_struct):
        # find length of chromosome
        length, min, max = algorithm.getParameters(parameter_struct)

        # initialize population chromosomes
        population = []
        # generate chromosomes in the population
        for chromosome in range(self.mu):
            chromo = []
            for weight_index in range(length):
                chromo.append(random.uniform(min, max))
            population.append(chromo)
        return population

    # evaluate the fitness of a group of chromosomes
    def evaluateGroupFitness(self, algorithm, parameter_struct, testing_set, group):
        group_fitness = []
        for chromosome_num in range(len(group)):
            group_fitness.append(algorithm.getFitness(testing_set, parameter_struct, group[chromosome_num]))
        return group_fitness

    # implements intermediate recombination
    def recombine(self, parents):
        offspring = []
        for gene in range(len(parents[0])):
            averaged_gene = 0
            for parent in range(len(parents)):
                averaged_gene += parents[parent][gene]
            averaged_gene /= len(parents)
            offspring.append(averaged_gene)
        return offspring

    # add gaussian noise to each gene of the offspring
    def mutate(self, chromosome):
        for gene in range(len(chromosome)):
            chromosome[gene] += random.gauss(0, self.sigma)

    def selection(self, population, offspring_group, population_fitness, offspring_group_fitness):
        next_generation = []
        next_generation_fitness = []
        group = population + offspring_group
        group_fitness = population_fitness + offspring_group_fitness
        #offspring_start = len(population)
        for chromo in range(self.mu):
            best_fitness_index = 0
            for index in range(1, len(group)):
                if group_fitness[index] > group_fitness[best_fitness_index]:
                    best_fitness_index = index
            next_generation.append(group[best_fitness_index])
            next_generation_fitness.append(group_fitness[best_fitness_index])
            if next_generation[-1] in offspring_group:
                self.probability_of_offspring += 1
            #if best_fitness_index >= offspring_start:
            #    self.probability_of_offspring += 1
            #else:
            #    offspring_start -= 1
            del group[best_fitness_index]
            del group_fitness[best_fitness_index]
        return next_generation, next_generation_fitness