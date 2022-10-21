from itertools import combinations
from random import sample
import os
import math
from typing import DefaultDict


data_sizes = [10,11,12,13,14,15]
range_left = 1 
range_right = 1000
I_MAX = range_right*range_right

#function to read data from file
def read_data(file_name, list_num):
    f = open(file_name,'r')
    num = int(f.readline())
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
    new_list = [[0 for a in range(row)] for i in range(row)]
    print(new_list)
    for i in range(row):
        for j in range(col):
            if (i==j):
                continue
            new_list[i][j] = math.dist(list_num[i],list_num[j])
    return new_list

#unnecessary
def nearest_city(distance_list):
    min = I_MAX
    for i,j in enumerate(distance_list):
        if j == 0:
            continue
        if j < min:
            min = i
            index = i
    distance_list[index] = 0
    return min,index     

def find_TSP(matrix):
    min = range_right*range_right
    rows = len(matrix)
    cols = len(matrix[0])
    sum = 0
    cnt = 0
    i = 0
    j = 0
    visited_point = DefaultDict(int)

    visited_point[0] = 1
    path = [0] * len(matrix)

    while i < rows and j < cols:
        if cnt >= cols - 1:
            break
        if j!= i and (visited_point[j] == 0):
            if matrix[i][j] < min:
                min = matrix[i][j]
                path[cnt] = j + 1

        j+=1

        if j == cols:
            sum += min
            min = range_right*range_right
            visited_point[path[cnt] - 1] = 1
            j = 0
            i = path[cnt] - 1
            cnt += 1

    i = path[cnt - 1] - 1

    for j in range(rows):
        if (i != j) and matrix[i][j] < min:
            min = matrix[i][j]
            path[cnt] = j + 1

    sum += min

    print("\n Cost: ", sum)

def main():
    tab = []
    num = read_data("text.txt", tab)
    #print(tab, num)
    #generate_data()
    
    new_list = edit_data(tab,num)
    print(new_list)
    find_TSP(new_list)
main()
