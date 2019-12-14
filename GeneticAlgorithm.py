import random
import copy
import MathAndStats as ms

class GeneticAlgorithm:

    def __init__(self, prob_cross, prob_mutation, mutation_variance, population_size, max_generations):
        self.prob_cross = prob_cross
        self.prob_mutation = prob_mutation
        self.mutation_variance = mutation_variance
        self.population_size = population_size
        self.max_generations = max_generations
        self.k = 2
        self.num_winning_pairs = 2

    def optimize(self, algorithm, parameter_struct, training_set):
        validation_index = int((float(len(training_set)) * 8 / 10)) - 1
        population = self.initializePopulation(algorithm, parameter_struct)
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[:validation_index], population)
        for generation in range(1, self.max_generations):
            print("GA: Working on generation", generation, "of", self.max_generations, "for", len(parameter_struct) - 1,
                  "layer MLP")
            print("Using: probability of crossing=", self.prob_cross, "probability of mutation=", self.prob_mutation,
                  "mutation variance", self.mutation_variance, "population size=", self.population_size)
            selection_group = self.selectChromosomes(population, population_fitness)
            recombined_group = self.recombine(selection_group)
            self.mutate(recombined_group)
            # get fitness for new chromosomes (recombined_group)
            recombined_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[:validation_index], recombined_group)
            population, population_fitness = self.replace(population, recombined_group, population_fitness, recombined_fitness)
            average_fitness = ms.getMean(population_fitness, len(population_fitness))
            print("GA: Average fitness for the generation: ", average_fitness)
        # evaluate the final population using the validation set to help fight overfitting
        population_fitness = self.evaluateGroupFitness(algorithm, parameter_struct, training_set[validation_index:], population)
        best_fitness_index = 0
        for index in range(1, len(population)):
            if population_fitness[index] > population_fitness[best_fitness_index]:
                best_fitness_index = index
        most_fit_chromosome = population[best_fitness_index]
        return most_fit_chromosome

    # initialize a population of chromosomes with random weights along a uniform distribution
    def initializePopulation(self, algorithm, parameter_struct):
        # find length of chromosome
        length, min, max = algorithm.getParameters(parameter_struct)

        # initialize population chromosomes
        population = []
        # generate chromosomes in the population
        for chromosome in range(self.population_size):
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

    # choose selection group using k tournament selection
    def selectChromosomes(self, population, population_fitness):
        selection_group = []
        # for loop iterating for the length of the desired selection group
        for pair in range(self.num_winning_pairs * 2):
            tournament_chromosomes = []
            tournament_chromosomes_fitnesses = []
            # get k random chromosomes from the population
            for i in range(self.k):
                selected_index = random.randint(0, len(population) - 1)
                tournament_chromosomes.append(population[selected_index])
                tournament_chromosomes_fitnesses.append(population_fitness[selected_index])
            # select the chromosome with the highest fitness and add it to the selection group
            lowest_index = 0
            for index in range(1, self.k):
                if tournament_chromosomes_fitnesses[index] > tournament_chromosomes_fitnesses[lowest_index]:
                    lowest_index = index
            selection_group.append(copy.deepcopy(tournament_chromosomes[lowest_index]))

        return selection_group

    # breed chromosomes using 2 parents: 2 children using prob_cross for probability of crossover
    def recombine(self, selection_group):
        recombined_group = []
        # while len(recombined_group) < len(selection_group):
        for pair in range(0, len(selection_group), 2):
            # get parents
            parents = []
            parents.append(selection_group[pair])
            parents.append(selection_group[pair + 1])
            # if we do crossover
            if random.uniform(0, 1) <= self.prob_cross:
                children = []
                children.append([])
                children.append([])
                for gene in range(len(parents[0])):
                    # if we take gene from parent 0
                    if random.uniform(0, 1) < 0.5:
                        children[0].append(parents[0][gene])
                        children[1].append(parents[1][gene])
                    else:
                        children[0].append(parents[1][gene])
                        children[1].append(parents[0][gene])
                recombined_group.append(children[0])
                recombined_group.append(children[1])
            # else add parents directly
            else:
                recombined_group.append(parents[0])
                recombined_group.append(parents[1])
        return recombined_group

    # mutate chromosomes of the next generation
    def mutate(self, recombined_group):
        for chromosome in range(len(recombined_group)):
            # if we mutate, add Gaussian noise to all genes
            if random.uniform(0, 1) <= self.prob_mutation:
                # get Gaussian noise with mean 0 and variance mutation_variance
                for gene in range(len(recombined_group[chromosome])):
                    recombined_group[chromosome][gene] += random.gauss(0, self.mutation_variance)

    # generate the next generation
    # add the most fit to the recombined group, then add from the previous population with replacement randomly
    def replace(self, population, recombined_group, population_fitness, recombined_fitness):
        # add most fit member to next generation
        best_fitness_index = 0
        for index in range(1, len(population)):
            if population_fitness[index] > population_fitness[best_fitness_index]:
                best_fitness_index = index
        recombined_group.append(population[best_fitness_index])
        recombined_fitness.append(population_fitness[best_fitness_index])
        print("Best fitness from previous population: ", recombined_fitness[-1])
        # add the rest of the previous generation randomly to fill out the next generation
        while len(recombined_group) < len(population):
            selected_index = random.randint(0, len(population) - 1)
            recombined_group.append(population[selected_index])
            recombined_fitness.append(population_fitness[selected_index])
        return recombined_group, recombined_fitness
