import math
import numpy as np
import random

'''
Best Solve.
Best Solution: 1.67099 at (-0.622866, - 0.827733)

            PROBLEM:
Find the Maximum point of a function.
x and y are between -2 and 2
Population size is 8
Crossover probability is 0.7
Mutation probability is 0.01
Number of generations is 200

Find (x, y) pair that gives the maximum point of the fitness function.
'''

MIN_RANGE = -2
MAX_RANGE = 2
MIN_BITS = 0 # 2^0
MAX_BITS = 255 # 2^8 - 1
VARIABLE_LENGTH = 8 # 8 bits
POPULATION_SIZE = 8 # 8 chromosomes of 8 bits must be even
CROSSOVER_PROBABILITY = 0.70 # 70%
MUTATION_PROBABILITY = 0.01 # 1%
NUMBER_OF_GENERATIONS = 200
MAX_CHROMOSOME_VALUE = 65535 # 2^16 - 1

def ConvertToBinary(chromosome):
    # Converts a decimal number to binary in string format
    return format(chromosome, '016b')

def ConvertToDecimal(chromosome):
    # Converts an 8 bit binary number in string format to decimal
    i = VARIABLE_LENGTH # 7 to 0
    decimal = 0
    for x in chromosome:
        i -= 1
        if (x == '1'):
            decimal += 2**i
    return decimal

def PutInRange(value):
    return (value * ((MAX_RANGE - MIN_RANGE) / MAX_BITS)) + MIN_RANGE

def SplitChromosome(chromosome):
    # Split the chromosome string in half
    return chromosome[:8], chromosome[8:]

def RouletteWheelSelection(Fitness):
    # Return a chromosome randomly (higher fitness has higher probability)
    # Return the index of the selected chromosome
    choice = random.randint(0, 99)
    total_fitness = 0
    for x in range(len(Fitness)):
        total_fitness += Fitness[x]
        if (total_fitness >= choice):
            return x

def Mutation(chromosome):
    # Randomly turn a bit in the chromosome and return it
    Mutate = random.random()
    if (Mutate > MUTATION_PROBABILITY):
        return chromosome # No mutation if greater than mutation probability
    x = random.randint(0, 15)
    chromosome = list(chromosome)
    if chromosome[x] == '0':
        chromosome[x] = '1'
    else:
        chromosome[x] = '0'
    return ''.join(chromosome)

def FitnessFunction(x, y):
    return ((1-x)**2 * math.e**(-x**2 - (y+1)**2)) - (x - x**3 - y**3) * math.e**(-x**2 - y**2)

def GetRandomChromosome():
    return random.randint(MIN_BITS, MAX_CHROMOSOME_VALUE)

def main():
    x_value = None
    y_value = None
    fitness = None
    # initial population
    Population = np.array([ConvertToBinary(GetRandomChromosome()) for x in range(POPULATION_SIZE)])
    for _ in range(NUMBER_OF_GENERATIONS):
        Fitness = np.empty(POPULATION_SIZE) # Fitness of each chromosome
        list_x = np.empty(POPULATION_SIZE) # x values of each chromosome
        list_y = np.empty(POPULATION_SIZE) # y values of each chromosome

        # Calculate fitness of each chromosome
        for x in range(len(Population)):
            a, b = SplitChromosome(Population[x])
            a = PutInRange(ConvertToDecimal(a))
            b = PutInRange(ConvertToDecimal(b))
            list_x[x] = a
            list_y[x] = b
            Fitness[x] = FitnessFunction(a, b)
        # Make the fitness values positive
        minFitness = Fitness.min()
        if (minFitness < 0): # If there are negative values
            minFitness = -minFitness + 0.001 # make it positive and add a small value to keep it >0
            Fitness += minFitness # Add it to all fitness values

        # Divide by SumOfFitnesses to get probability then Multiply by 100 to get percentage of probability
        Fitness = (Fitness / Fitness.sum()) * 100

        # mating
        NextGeneration = np.empty_like(Population)
        for x in range(0, POPULATION_SIZE, 2): # 2 offspring per couple
            parent1 = RouletteWheelSelection(Fitness) # index of parent1
            parent2 = RouletteWheelSelection(Fitness) # index of parent2
            crossOver = random.random()
            if (crossOver < CROSSOVER_PROBABILITY):
                cut = random.randint(1, 15) # cut point
                NextGeneration[x] = Mutation(Population[parent1][:cut] + Population[parent2][cut:]) # Offspring1
                cut = random.randint(1, 15) # cut point
                NextGeneration[x+1] = Mutation(Population[parent2][:cut] + Population[parent1][cut:]) # Offspring2
            else:
                NextGeneration[x] = Mutation(Population[parent1]) # clone1
                NextGeneration[x+1] = Mutation(Population[parent2]) # clone2
        if (_ == NUMBER_OF_GENERATIONS - 1): # last generation
            index = np.argmax([FitnessFunction(list_x[x], list_y[x]) for x in range(POPULATION_SIZE)])
            x_value = list_x[index]
            y_value = list_y[index]
            fitness = FitnessFunction(list_x[index], list_y[index])
        Population = NextGeneration # set Population to NextGeneration
    print(f"Best x value: {x_value}")
    print(f"Best y value: {y_value}")
    print(f"Best fitness: {fitness}")
main()