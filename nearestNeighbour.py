import math
import random

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

    solution = []
    visited = []
    best_dist = 0

    if r:
        current = l[random.randint(0,len(l))-1]
    else:
        current = l[0]

    solution.append(current)
    visited.append(current)

    while len(visited) < len(l):
        closest_dist = 999999999
        closest = 999999999

        for v in l:
            if v in visited:
                continue
            dist = getDistance(current,v)
            if dist < closest_dist:
                closest_dist = dist
                closest = v

        solution.append(closest)
        visited.append(closest)
        current = closest
        best_dist +=closest_dist

    best_dist += getDistance(solution[0], solution[len(solution)-1])
    solution.append(solution[0])

    #print("solution: {}".format(solution))
    #print("distance: {}".format(best_dist))

    return best_dist







