import math
import random
import globals

def loadData(tab):
    V = len(tab)

    l = []
    for i in range(V):
        x = tab[i][0]
        y = tab[i][1]
        l.append((x,y))
    return l

def getDistance(v1,v2):
    dx = int(v2[0]) - int(v1[0])
    dy = int(v2[1]) - int(v1[1])
    return math.sqrt((math.pow(dx,2)+math.pow(dy,2)))

def nearestN(l,r):
    #wybierz losowy wierzcholek jako aktualny i odwiedzony
    #znajdz najblizszy wierzcholek
    #dodaj do rozwiazania krawedz laczaca aktualny i znaleziony wierzcholek
    #ustaw znaleziony jako aktualny i odwiedzony
    #powtorz jesli istnieja nieodwiedzone wierzcholki
    #do rozwiazania dodaj krawedz laczaca ostatni z pierwszym

    new_l = []
    num = 0
    for i in range(len(l)):
        new_l.append((l[i],i))

    solution = []

    visited = []
    best_dist = 0

    if r:
        current = new_l[random.randint(0,len(l))-1][0]
    else:
        current = new_l[0][0]

    solution.append(current)
    visited.append(current)

    while len(visited) < len(l):
        closest_dist = 999999999
        closest = 999999999

        for v in new_l:
            if v[0] in visited:
                continue
            dist = getDistance(current,v[0])
            if dist < closest_dist:
                closest_dist = dist
                closest = v[0]
                num = v[1]

        solution.append(closest)
        visited.append(closest)

        globals.nn_solution.append(num)

        current = closest
        best_dist +=closest_dist


    globals.nn_solution.append(globals.nn_solution[0])
    best_dist += getDistance(solution[0], solution[len(solution)-1])
    solution.append(solution[0])

    #print("solution: {}".format(solution))
    #print("distance: {}".format(best_dist))

    return best_dist







