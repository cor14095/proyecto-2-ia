##
## emgmm.py
##
## By: Alejandro 'Perry' Cortes
##
## This script categorize data based on gassians
##

## Imports

import re
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal as mn

# Globals
sigma = []
mu = []
pi = []

## Functions
def readFile(pathToFile):
    np_array = np.empty(shape=[0, 2])
    array = []
    with open(pathToFile, 'r') as file:
        for line in file:
            line = re.sub('[\]\[]', '', line)
            points = line.split(',')
            array.append([float(points[0]), float(points[1])])
        np_array = np.asarray(array)
    return np_array

def gaussian(sigmaI, muI, x):
    # pdf(x, mean=None, cov=1)	Probability density function.
    return mn.pdf(x=x, mean=muI, cov=sigmaI, allow_singular=True)

def updateGamma(sigmaI, muI, piI, x, clusteSize):
    global sigma
    global mu
    global pi
    # Para cada cluster i
    normal = gaussian(sigmaI, muI, x) # Obtenemos la normal.
    e_ij = normal * piI    # El e sub ji.
    R = sum([(gaussian(sigma[w], mu[w], x) * pi[w])
             for w in range(0, clusteSize)])
        #print("Loop {} of {}".format(j + 1, clusteSize))
    # eij = eij / R
    # Si R es muy pequeno se vuelve infinito
    return (e_ij/R) if (R >= 0) else (e_ij)

def nextGamma(sigmaI, muI, piI, data, clusteSize):
    return sum([updateGamma(sigmaI, muI, piI, point, clusteSize) for point in data])

def nextSigma(prevMU, data):
    return np.cov(np.append(data, [prevMU], axis=0).T)

def nextMU(gamma, prevSigma, prevMU, prevPI, data, i, clusteSize):
    updatedMU = 0
    for value in data:
        updatedMU = np.dot(
            updateGamma(prevSigma, prevMU, prevPI, value, clusteSize),
            value
        )
    return updatedMU / gamma

def expectation(data, clusteSize):
    global sigma
    global mu
    global pi

    cluster_expectancy = []
    for point in data:
        expectancies_ = [updateGamma(sigma[w], mu[w], pi[w], point, clusteSize) for w in range(clusteSize)]
        index_ = expectancies_.index(max(expectancies_))
        cluster_expectancy.append(index_)
    return cluster_expectancy

def eigenso(covariance_):
    values_, vectors_ = np.linalg.eigh(covariance_)
    order = values_.argsort()[::-1]
    return values_[order], vectors_[:, order]

## main
def main():
    global sigma
    global mu
    global pi

    # Variables
    gamma = 0
    iterations = 0

    # Leer la data
    data = readFile("test_gmm_1.txt")
    closterAmount = int(input("Ingrese la cantidad de clusters:\n-> "))

    mu = np.random.uniform(0, 9, size=(closterAmount, 2))

    # Inicializacion
    for x in range(closterAmount):
        mu[x] = data[random.randint(0, len(data))]

    # K gaussianos aleatorios
    for x in range(0, closterAmount):
        sigma.append(np.cov(data.T))
    # K pi para cada cluster
    pi = [1. / closterAmount] * closterAmount

    print(mu)
    # Mientras los gaussianos se muevan o llegue al limite de iteraciones.
    while iterations < 100:
        # Paso E
        for i in range(0, closterAmount):
            # Por cada dato de entrada x
            gamma = nextGamma(sigma[i], mu[i], pi[i], data, closterAmount)
            # Tengo que hacer copias para evitar que python use las originales
            copyMU = copy.copy(mu[i])
            copyPI = copy.copy(pi[i])
            copySigma = copy.copy(sigma[i])
            # Paso M
            mu[i] = nextMU(gamma, copySigma, copyMU, copyPI, data, i, closterAmount)
            sigma[i] = nextSigma(mu[i], data)
            pi[i] = gamma / len(data)
        iterations += 1

    print (mu)

    # Ahora vamos a mostrar los gauseanos
    try:
        fig, ax = plt.subplots()

        probs = expectation(data, closterAmount)
        colors = "bgrcmykw"

        for i in range(len(data)):
            [xp, yp] = data[i]
            ax.scatter(xp, yp, color=colors[probs[i]])

        for i in range(0, closterAmount):
            [xm, ym] = mu[i]
            covariance = sigma[i]
            values, vectors = eigenso(covariance)
            th = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
            for j in range(0, 5):
                width, height = j * np.sqrt(values)
                elipse = Ellipse(xy=(xm, ym), width=width, height=height, angle=th, color=colors[i])
                elipse.set_facecolor('none')
                ax.add_artist(elipse)
            ax.scatter(xm, ym, marker='^', color=colors[i])

        plt.show()
    except:
        print("Something whent wrong while plotting...")

    try:
        ex = input("quit to exit, point as \'x,y\'")
        while ex != "quit":
            [x, y] = ex.split(',')
            new_point = np.array([[x, y]])
            expectancies = [updateGamma(sigma[k], mu[k], pi[k], new_point, closterAmount) for k in range(closterAmount)]
            index = expectancies.index(max(expectancies))
            print("Belongs to cluster ", index)
            print(expectancies)
            ex = input("quit to exit, point as \'x,y\'")
    except:
        print("Quit program...")

## Program

if __name__ == "__main__":
    # execute only if run as a script
    main()
