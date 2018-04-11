##
## emgmm.py
##
## By: Alejandro 'Perry' Cortes
##
## This script categorize data based on gassians
##

## Imports

import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal as mn

## Functions
def readFile(pathToFile):
    points = []
    with open(pathToFile, 'r') as coordsFile:
        for line in coordsFile:
            cleanLine = ""
            # In case the format is [#, #]
            for character in line:
                if (character != '[' and character != ']' and character != ' '):
                    cleanLine += character
            values = cleanLine.split(',')
            points.append([float(values[0]), float(values[1])])
        coordinates = np.asarray(points)
    return coordinates

def gaussian(sigma, mu, x):
    # pdf(x, mean=None, cov=1)	Probability density function.
    return mn.pdf(x, mu, sigma)

def updateGamma(sigma, mu, pi, x, dataSize, i, clusterSize):
    # Para cada cluster i
    normal = gaussian(sigma[i], mu[i], x) # Obtenemos la normal.
    e_ij = normal * pi[i] / dataSize    # El e sub ji.
    R = 0
    for j in range(0, clisteSize):
        R += gaussian(sigma[j], mu[j], x) * pi[j] / dataSize
    # eij = eij / R
    # Si R es muy pequeno se vuelve infinito
    return (e_ij/R) if (R > 0) else (e_ij)

def nextGamma(sigma, mu, pi, data, i, closterAmount):
    nextGamma = 0
    # for point in data:
    #     nextGamma +=
    return 0

def nextSigma(gamma, prevSigma, prevMU, prevPI):
    return 0

def nextMU(gamma, prevSigma, prevMU, prevPI):
    return 0

def nextPi(gamma):
    return 0

## main
def main():

    # Variables
    sigma = []
    mu = []
    pi = []
    gamma = 0
    iterations = 0

    # Leer la data
    data = readFile("coordinates.txt")
    closterAmount = int(input("Ingrese la cantidad de clusters:\n-> "))

    mu = np.random.uniform(0, 9, size=(closterAmount, 2))

    # Inicializacion
    # K gaussianos aleatorios
    for x in range(0, closterAmount):
        rand = random.uniform(0, 9)
        sigma.append(np.array([[rand, 0], [0, rand]]))
    # K pi para cada cluster
    pi = [1. / closterAmount] * closterAmount

    # Mientras los gaussianos se muevan o llegue al limite de iteraciones.
    while iterations < 100:
        # Paso E
        for i in range(0, closterAmount):
            # Por cada dato de entrada x
            gamma = nextGamma(sigma, mu, pi, data, i, closterAmount)
            # Paso M
            mu[i] = nextMU(gamma, sigma[i], mu[i], pi[i])
            sigma[i] = nextSigma(gamma, sigma[i], mu[i], pi[i])
            pi[i] = nextPi(gamma)
        iterations += 1

## Program

if __name__ == "__main__":
    # execute only if run as a script
    main()
