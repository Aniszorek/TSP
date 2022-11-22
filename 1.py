from itertools import combinations
from random import sample
import os
import math
import numpy as np

is_data_numbered = True # if data in files is like "num x y"
data_sizes = [100, 150]
range_left = 1 
range_right = 1000
I_MAX = range_right*range_right

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
    new_list = np.empty((row,col))
    #print(new_list)
    for i in range(row):
        for j in range(col):
            if (i==j):
                new_list[i][j] = np.inf
                continue
            new_list[i][j] = math.dist(list_num[i],list_num[j])
    return new_list


def main():
    tab = []
    num = read_data("250TSP.txt", tab)
    distances = edit_data(tab,num)
    print(distances, num)
    print(np.ones(distances.shape) / distances)
    #generate_data()
    #distances = np.array(tab)
    #print(distances)


main()