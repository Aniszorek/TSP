import random
from itertools import combinations
from random import sample
import os
import math
import numpy as np
import nearestNeighbour as nn
import time
import matplotlib.pyplot as plt

import globals as g

is_data_numbered = True  # if data in files is like "num x y"
data_sizes = [5, 100, 150]
range_left = 1
range_right = 1000
I_MAX = range_right * range_right


#####################################  SETTINGS  #################################################

FILEPATH = "tsp1000.txt"
NUMBER_OF_ANTS = 10
NUM_OF_ITERATIONS = 200000
VAPORIZATION = 0.3
Q = 0.9          #smallQ - more aco algo Q=1 - only greedy - go to vertex with most pheromone
TIME_LIMIT = 300  # seconds
ALFA = 13.37 # best so far 13.67
BETA = 2.137
USE_GREEDY_AS_STARTPOINT = True #should phermone level be initialized with greedy algorithm path?

LOCAL_SEARCH = False        #local search on each feasible solution
LOCAL_FOR_IT_BEST = False   #ls only on iteration best
LOCAL_FOR_BEST = False      #initial ls on greedy solution - pheromone lvl initialized with path found by this algorithm - good for dense graphs

GREEDY_MULTIPLIER = 500      #good is number of cities in graph

##############################################################################################

BIG_NUMBER = 99999999

#DEBUG
PRINT_BEST = True
PRINT_IT_BEST = True
PRINT_TIME = True


# function to read data from file
def read_data(file_name, list_num):
    f = open(file_name, 'r')
    num = int(f.readline())
    if is_data_numbered:
        for i in range(num):
            list_num.append(list(map(float, f.readline().split()[1:3])))
    else:
        for i in range(num):
            list_num.append(list(map(int, f.readline().split())))
    f.close()
    return num


# function to generate numbers in the given range [range_left and range_right]
def generate_data():
    try:
        os.mkdir('tests')
    except:
        pass
    for size in data_sizes:
        f_out = open('{}\{}.in'.format('tests', size), 'w')
        f_out.write(str(size) + '\n')
        nums = sample(list(combinations(range(range_left, range_right), 2)), size)
        for i in nums:
            # print(num)
            f_out.write(str(i[0]) + ' ' + str(i[1]) + '\n')


# function to generate matrix of distance between point
def edit_data(list_num, row):
    col = row
    new_list = np.empty([row, col])
    for i in range(row):
        for j in range(col):
            if i == j:
                new_list[i][j] = -1
                continue
            new_list[i][j] = math.dist(list_num[i], list_num[j])
    return new_list


greedy_value = 0
plot_data = []

def main():

    #initialize global variables
    g.init()

    global plot_data

    tab = []
    # generate_data()
    num = read_data(FILEPATH, tab)

    plot_data = np.copy(tab)


    global greedy_value
    tab = nn.loadData(tab)
    greedy_value = nn.nearestN(tab, True)

    #create matrix of distances between each pair of points
    distances = edit_data(tab, num)

    #pheromone = np.ones(distances.shape) / distances
    pheromone = np.ones(distances.shape) * (1 / (greedy_value * GREEDY_MULTIPLIER))
    np.fill_diagonal(pheromone, 0)


    #initialize pheromone level based on greedy solution
    if USE_GREEDY_AS_STARTPOINT:
        for i in range(0, len(g.nn_solution) - 1):
            Update_pheromone(g.nn_solution[i], g.nn_solution[i + 1], distances, pheromone, True, greedy_value)


    probability = np.ones(distances.shape)
    probability = probability * (1 / (greedy_value * GREEDY_MULTIPLIER))
    np.fill_diagonal(probability, 0)

    print("greedy {}".format(greedy_value))
    print("ACS {}".format(ACS(distances, probability, pheromone)))
    print("greedy {}".format(greedy_value))

def ACS(distances, probability, pheromone):
    global TIME_LIMIT
    global NUM_OF_ITERATIONS
    global BIG_NUMBER
    global NUMBER_OF_ANTS
    global Q
    global LOCAL_SEARCH

    row = len(distances)

    best_solution = g.nn_solution
    best_distance = CalculateSolutionValue(best_solution,distances)

    current_iteration = 0
    initialized = False

    while NUM_OF_ITERATIONS > current_iteration and TIME_LIMIT > 0:
        timer_start = time.perf_counter()

        # [DEBUG] iteration best
        it_best = 99999999
        it_best_sol = []

        # LOCAL SEARCH ON BEST SOLUTION:
        # apply local search (2-opt) and if found better solution, then update pheromone again
        #done only at first iteration
        if LOCAL_FOR_BEST and initialized is False:
            print("Wstepny local search")
            local_search_sol = []
            update_pheromone_again = False
            for i in range(len(best_solution) - 2):

                for j in range(i + 1, len(best_solution) - 1):
                    local_search_sol = localSearch(best_solution, i, j, distances)
                    if local_search_sol:
                        local_search_distance = CalculateSolutionValue(local_search_sol, distances)
                        if local_search_distance < best_distance:
                            best_solution = local_search_sol
                            best_distance = local_search_distance
                            update_pheromone_again = True
                            print("found better local: {}".format(best_distance))


            # add pheromone as ant would go this path
            if update_pheromone_again:
                for i in range(len(best_solution) - 1):
                    Update_pheromone(best_solution[i], best_solution[i + 1], distances, pheromone,True, best_distance)

        #exp2 - initialize pheromone based on best local
        if LOCAL_FOR_BEST and initialized is False:
                if initialized is False:
                    initialized = True
                    for i in range(0, len(best_solution) - 1):
                        Update_pheromone(best_solution[i], best_solution[i + 1], distances, pheromone, True,best_distance)


        current_iteration += 1
        print("Current iteration {}".format(current_iteration))
        for i in range(NUMBER_OF_ANTS):
            start = random.randrange(0, row - 1)
            current_sol = [start]
            ant_probability = np.copy(probability)

            for i in range(row):
                ant_probability[i][start] = 0

            while len(current_sol) < row:
                # State transition rule
                q = random.uniform(0, 1)
                if q <= Q:
                    # exploitation
                    best_pheromone = 0
                    for i in range(row):
                        if ant_probability[current_sol[-1]][i] == 0:
                            continue
                        cur_pheromone = (pheromone[current_sol[-1]][i]) * ((1 / distances[current_sol[-1]][i]) ** BETA)
                        if cur_pheromone > best_pheromone:
                            best_pheromone = cur_pheromone
                            next = i

                else:
                    # exploration
                    #fill antProbability matrix depending on pheromone and distances between points
                    ant_probability = Calculate_probability(current_sol[-1], row, ant_probability, pheromone, distances)
                    #choose next point with given probability - ant_probability[current_sol[-1]] - row with probabilities of moving from last visited to each next possible point
                    next = random.choices([x for x in range(row)], ant_probability[current_sol[-1]])[0]

                #set probabilities of returning to visited point to 0
                for i in range(row):
                    ant_probability[i][next] = 0

                # update pheromone
                Update_pheromone(current_sol[-1], next,distances, pheromone)

                #add new point to solution
                current_sol.append(next)

            #END OF WHILE

            #get value of current solution
            distance = CalculateSolutionValue(current_sol, distances)

            if distance < it_best:
                it_best = distance
                it_best_sol = current_sol

            # apply local search (2-opt) and if found better solution, then update pheromone again
            if LOCAL_SEARCH:
                local_search_sol = []
                update_pheromone_again = False
                for i in range(len(current_sol) - 2):
                    if update_pheromone_again:
                        break
                    for j in range(i + 1, len(current_sol) - 1):
                        local_search_sol = localSearch(current_sol, i, j, distances)
                        if local_search_sol:
                            local_search_distance = CalculateSolutionValue(local_search_sol, distances)
                            if local_search_distance < best_distance:
                                current_sol = local_search_sol
                                distance = local_search_distance
                                update_pheromone_again = True
                                print("found better local: {}".format(distance))
                        if update_pheromone_again:
                            break

                if update_pheromone_again:
                    # add pheromone as ant would go this path
                    for i in range(len(current_sol) - 1):
                        Update_pheromone(current_sol[i], current_sol[i + 1],distances, pheromone)

            #check if current sol is best overall
            if distance < best_distance:
                best_solution = current_sol
                best_distance = distance


        # GLOBAL PHEROMONE UPDATE

        #update pheromone on best path found yet
        for i in range(0, len(best_solution) - 2):
            Update_pheromone(best_solution[i], best_solution[i + 1],distances, pheromone, True, best_distance)
        Update_pheromone(best_solution[0], best_solution[-1],distances, pheromone, True, best_distance)




        # LOCAL SEARCH ON ITERATION BEST SOLUTION:
        if LOCAL_FOR_IT_BEST:
            print("local search on iteration best")
            local_search_sol = []
            update_pheromone_again = False
            for i in range(len(it_best_sol) - 2):

                for j in range(i + 1, len(it_best_sol) - 1):
                    local_search_sol = localSearch(it_best_sol, i, j, distances)
                    if local_search_sol:
                        local_search_distance = CalculateSolutionValue(local_search_sol, distances)
                        if local_search_distance < best_distance:
                            best_solution = local_search_sol
                            it_best_sol = local_search_sol
                            best_distance = local_search_distance
                            it_best = local_search_distance
                            update_pheromone_again = True
                            print("found better local: {}".format(best_distance))


            if update_pheromone_again:
                #add pheromone as ant would go this path
                for i in range(len(best_solution) - 1):
                    Update_pheromone(best_solution[i], best_solution[i + 1], distances, pheromone,True, best_distance)


        timer_end = time.perf_counter()

        TIME_LIMIT -= timer_end - timer_start
        if PRINT_TIME:
            print("time: {}, left: {}".format(timer_end - timer_start,TIME_LIMIT))
        if PRINT_BEST:
            print("best {}".format(best_distance))
        if PRINT_IT_BEST:
            print("iteration best {}".format(it_best))
        print("------------------------------------")

    # sanity check
    print(best_solution)
    rep = []
    for el in best_solution:
        if best_solution.count(el) > 1:
            if el not in rep:
                print("Repeated number: {}".format(el))
    print("dlugosc rozwiazania")
    print(len(best_solution))

    best_distance = CalculateSolutionValue(best_solution,distances)

    createPlot(best_solution,distances)
    return best_distance


def Calculate_probability(i, row, probability, pheromone, distances):
    p = 0
    numerator = 0
    denominator = 0
    #p[i][j]^A * (1/dst[i][j])^B / SUMA po mozliwych polaczeniach: p[i][j]^A * (1/dst[i][j])^B
    # calculate denominator
    for j in range(row):
        if probability[i][j] == 0:
            continue
        denominator = denominator + np.power(pheromone[i][j], ALFA) * np.power(1 / distances[i][j], BETA)

    for j in range(row):
        if probability[i][j] == 0:
            continue
        numerator = np.power(pheromone[i][j], ALFA) * np.power(1 / distances[i][j], BETA)
        p = numerator / denominator
        probability[i][j] = p

    return probability


def Update_pheromone(i, j,distances, pheromone, offline=False, best=0):

    if offline == False:
        #local update
        p_initial = VAPORIZATION * (1 / (greedy_value * GREEDY_MULTIPLIER))
        p_new = (1 - VAPORIZATION) * pheromone[i][j]
        pheromone[i][j] = p_initial + p_new
    else:
        #global for best
        p_1 = (1 - VAPORIZATION) * pheromone[i][j]
        p_2 = VAPORIZATION * (1 / best)
        pheromone[i][j] = p_1 + p_2

def globalPheromone(i,j,distances,pheromone):
    #global for other than best
    pheromone[i][j] = (1-VAPORIZATION)*pheromone[i][j]

def CalculateSolutionValue(solution, distances):
    distance = 0
    for i in range(0, len(solution) - 1):
        distance += distances[solution[i]][solution[i + 1]]
    distance += distances[solution[0]][solution[-1]]

    return distance


def localSearch(solution, i, j, distances):
    if -distances[i][i + 1] - distances[j][j + 1] + distances[i + 1][j + 1] + distances[i][j] < 0:
        better_sol = []
        better_sol.extend(solution[0:i + 1])
        reverse = solution[i + 1:j + 1]
        better_sol.extend(reverse[::-1])
        better_sol.extend(solution[j + 1:])
        return better_sol

    return []

def createPlot(solution,distances):

    global plot_data
    x = []
    y = []
    for i in range(len(solution)):
        x.append(plot_data[solution[i]][0])
        y.append(plot_data[solution[i]][1])
    x.append(plot_data[solution[0]][0])
    y.append(plot_data[solution[0]][1])
    plt.plot(x,y,'bo',linestyle="-")
    plt.show()


main()
