import random
from itertools import combinations
from random import sample
import os
import math
import numpy as np
import nearestNeighbour as nn
import time
import matplotlib.pyplot as plt

is_data_numbered = True  # if data in files is like "num x y"
data_sizes = [5, 100, 150]
range_left = 1
range_right = 1000
I_MAX = range_right * range_right

FILEPATH = "qa194.tsp"

NUMBER_OF_ANTS = 10
NUM_OF_ITERATIONS = 200000000
VAPORIZATION = 0.05
Q = 0.9
TIME_LIMIT = 300  # seconds
ALFA = 13.67    # best so far 13.67
BETA = 2.137

LOCAL_SEARCH = True

# used in update pheromone
GREEDY_MULTIPLIER = 194  # good is number of ncities

BIG_NUMBER = 99999999


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
    global plot_data
    tab = []
    # generate_data()
    num = read_data(FILEPATH, tab)

    plot_data = np.copy(tab)

    distances = edit_data(tab, num)

    probability = np.ones(distances.shape)
    # print(probability)

    # print(distances, num)
    # distances = np.array(tab)
    # print(distances)
    pheromone = np.ones(distances.shape) / distances
    np.fill_diagonal(pheromone, 0)

    tab = nn.loadData(tab)
    global greedy_value
    greedy_value = nn.nearestN(tab, True)

    probability = probability * (1 / greedy_value * GREEDY_MULTIPLIER)
    np.fill_diagonal(probability, 0)

    print("greedy {}".format(greedy_value))
    print("ACS {}".format(ACS(distances, probability, pheromone)))


def ACS(distances, probability, pheromone):
    global TIME_LIMIT
    global NUM_OF_ITERATIONS
    global BIG_NUMBER
    global NUMBER_OF_ANTS
    global Q
    global LOCAL_SEARCH

    row = len(distances)

    best_distance = BIG_NUMBER
    best_solution = []



    current_iteration = 0

    while NUM_OF_ITERATIONS > current_iteration and TIME_LIMIT > 0:
        timer_start = time.perf_counter()
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
                    Calculate_probability(current_sol[-1], row, ant_probability, pheromone, distances)
                    # print(ant_probability[current_sol[-1]])
                    next = random.choices([x for x in range(row)], ant_probability[current_sol[-1]])[0]
                    # print(next)

                for i in range(row):
                    ant_probability[i][next] = 0

                current_sol.append(next)

                # update feromonu
                Update_pheromone(current_sol[-1], next, pheromone)

                # print("CURRENT SOL: ")
                # print(current_sol)

            distance = CalculateSolutionValue(current_sol, distances)

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
                        Update_pheromone(current_sol[i], current_sol[i + 1], pheromone)

            if distance < best_distance:
                best_solution = current_sol
                best_distance = distance

        # global pheromone update
        edges = []
        for i in range(0, len(best_solution) - 2):
            Update_pheromone(best_solution[i], best_solution[i + 1], pheromone, True, best_distance)
            edges.append([best_solution[i], best_solution[i + 1]])
        Update_pheromone(best_solution[0], best_solution[-1], pheromone, True, best_distance)

        # update feromonu krawedzi, ktore nie sa w best solution
        for i in range(row):
            for j in range(row):
                needUpdate = True
                for edge in edges:
                    if i == edge[0] and j == edge[1]:
                        needUpdate = False
                if needUpdate:
                    pheromone[i][j] = (1 - VAPORIZATION) * pheromone[i][j] + VAPORIZATION * (
                            1 / (greedy_value * GREEDY_MULTIPLIER))

        timer_end = time.perf_counter()

        TIME_LIMIT -= timer_end - timer_start
        print("time: {}".format(timer_end - timer_start))
        print("best {}".format(best_distance))

    # sanity check
    print(best_solution)
    rep = []
    for el in best_solution:
        if best_solution.count(el) > 1:
            if el not in rep:
                print("Repeated number: {}".format(el))

    createPlot(best_solution,distances)

    return best_distance


def Calculate_probability(i, row, probability, pheromone, distances):
    p = 0
    numerator = 0
    denominator = 0

    # print("CALCULATE")

    # calculate denominator
    for j in range(row):
        # print(probability[i][j])
        if probability[i][j] == 0:
            continue
        denominator = denominator + np.power(pheromone[i][j], ALFA) * np.power(1 / distances[i][j], BETA)

    for j in range(row):
        if probability[i][j] == 0:
            continue
        numerator = np.power(pheromone[i][j], ALFA) * np.power(1 / distances[i][j], BETA)
        p = numerator / denominator
        probability[i][j] = p


def Update_pheromone(i, j, pheromone, offline=False, best=0):
    # print("UPDATE")

    if offline == False:

        # print(i)

        p_initial = VAPORIZATION * (1 / (greedy_value * GREEDY_MULTIPLIER))
        p_new = (1 - VAPORIZATION) * pheromone[i][j]
        pheromone[i][j] = p_initial + p_new
    else:
        p_1 = (1 - VAPORIZATION) * pheromone[i][j]
        p_2 = VAPORIZATION * (1 / best)
        pheromone[i][j] = p_1 + p_2


def CalculateSolutionValue(solution, distances):
    distance = 0
    for i in range(0, len(solution) - 2):
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
