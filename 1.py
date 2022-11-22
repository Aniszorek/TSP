import random
from itertools import combinations
from random import sample
import os
import math
import numpy as np
import nearestNeighbour as nn
import sys

is_data_numbered = True # if data in files is like "num x y"
data_sizes = [5,100, 150]
range_left = 1 
range_right = 1000
I_MAX = range_right*range_right

NUMBER_OF_ANTS = 20
NUM_OF_ITERATIONS = 10
VAPORIZATION = 0.9

ALFA = 1
BETA = 1

#used in update pheromone
GREEDY_MULTIPLIER = 1

BIG_NUMBER = 99999999

#function to read data from file
def read_data(file_name, list_num):
    f = open(file_name,'r')
    num = int(f.readline())
    if is_data_numbered:
        for i in range(num):
            list_num.append(list(map(int,f.readline().split()[1:3])))
    else:
        for i in range(num):
            list_num.append(list(map(int,f.readline().split())))
    f.close()
    return num

#function to generate numbers in the given range [range_left and range_right]
def generate_data():
    try:
        os.mkdir('tests')
    except:
        pass
    for size in data_sizes:
        f_out = open('{}\{}.in'.format('tests', size), 'w')
        f_out.write(str(size) + '\n')
        nums = sample(list(combinations(range(range_left,range_right),2)),size)
        for i in nums:
            #print(num)
            f_out.write(str(i[0]) + ' ' + str(i[1]) + '\n')

#function to generate matrix of distance between point
def edit_data(list_num, row):
    col = row
    new_list = np.empty([row,col])
    for i in range(row):
        for j in range(col):
            if (i==j):
                new_list[i][j] = -1
                continue
            new_list[i][j] = math.dist(list_num[i],list_num[j])
    return new_list


greedy_value = 0

def main():
    tab = []
    #generate_data()
    num = read_data("250TSP.txt", tab)
    distances = edit_data(tab,num)
    probability = np.ones(distances.shape)
    #print(probability)

    #print(distances, num)
    #distances = np.array(tab)
    #print(distances)
    pheromone = np.ones(distances.shape) / distances
    np.fill_diagonal(pheromone,0)

    tab = nn.loadData(tab)
    global greedy_value
    greedy_value = nn.nearestN(tab,True)

    probability = probability * (1/greedy_value*GREEDY_MULTIPLIER)
    np.fill_diagonal(probability,0)

    print("greedy {}".format(greedy_value))
    print("ACS {}".format(ACS(distances,probability,pheromone)))



def ACS(distances, probability, pheromone):

    solutions = []
    row = len(distances)

    best_distance = BIG_NUMBER
    best_solution = []

    iterations = 10

    while iterations > 0:
        iterations -= 1
        for i in range(NUMBER_OF_ANTS):
            #wybierz wierzcholek startowy
            #dopoki nie odwiedzono wszystkich
                #wylosuj kolejny wierzcholek
                #dodaj go do aktualnego rozwiazania
            #wybierz najlepsze rozwiazanie
            #offline update feromonu dla najlepszego rozwiazania
            start = random.randrange(0,row-1)
            current_sol = []
            current_sol.append(start)
            ant_probability = np.copy(probability)

            for i in range(row):
                ant_probability[i][start] = 0


            while len(current_sol) < row:
                Calculate_probability(current_sol[-1], row,ant_probability,pheromone,distances)
                next = random.choices([x for x in range(row)], ant_probability[current_sol[-1]])[0]

                #print(next)

                for i in range(row):
                    ant_probability[i][next] = 0

                current_sol.append(next)


                #update feromonu
                Update_pheromone(current_sol[-1],next,pheromone)

                #print("CURRENT SOL: ")
                #print(current_sol)

            distance = CalculateSolutionValue(current_sol,distances)

            if distance < best_distance:
                best_solution = current_sol
                best_distance = distance

        for i in range(0, len(best_solution) - 2):
            Update_pheromone(best_solution[i], best_solution[i + 1],pheromone, True, best_distance)
        Update_pheromone(best_solution[0], best_solution[-1],pheromone, True, best_distance)

    return best_distance




def Calculate_probability(i,row,probability,pheromone,distances):
    p = 0;
    numerator =0
    denominator = 0

    #print("CALCULATE")

    #calculate denominator
    for j in range(row):
        #print(probability[i][j])
        if probability[i][j] == 0:
            continue
        denominator = denominator + np.power(pheromone[i][j],ALFA) * np.power(1/distances[i][j],BETA)

    for j in range(row):
        if probability[i][j] == 0:
            continue
        numerator = np.power(pheromone[i][j],ALFA) * np.power(1/distances[i][j],BETA)
        p = numerator/denominator
        probability[i][j] = p

def Update_pheromone(i,j,pheromone,offline = False, best = 0):

    #print("UPDATE")


    if offline == False:

        #print(i)

        p_initial = VAPORIZATION * (1/(greedy_value*GREEDY_MULTIPLIER))
        p_new = (1 - VAPORIZATION) * pheromone[i][j]
        pheromone[i][j] = p_initial + p_new
    else:
        p_1 = (1- VAPORIZATION)* pheromone[i][j]
        p_2 = VAPORIZATION * (1/best)
        pheromone[i][j] = p_1 + p_2


def CalculateSolutionValue(solution,distances):
    distance = 0
    for i in range(0,len(solution) -2):
        distance += distances[solution[i]][solution[i+1]]
    distance += distances[solution[0]][solution[-1]]


    return distance







main()