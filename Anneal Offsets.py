#=================================================IMPORTS=========================================================#
print("Started Program...")
import dwave
import dimod
import math
import json
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from dwave.system.samplers import DWaveSampler
from dwave.embedding.pegasus import find_clique_embedding
#from pandas import DataFrame #We can delete this after the pandas stuff has been translated over to numpy/regular python
import numpy as numpy #it's been a long day 

token = "put your token here :)"
computer = "Advantage_system1.1"

#================================================JSON READER=====================================================#

print("Started reading json file...")
inputFile = [[0.3576734259547776, [[0.3578421714920063, 0.0003333333333333334], [0.35763199025343234, 0.0003333333333333334]]], [0.43751589332926805, [[0.4378207965666029, 0.0005833333333333333], [0.4375708301205863, 0.0004166666666666667]]]]

trackList = [] #This is just a big list of the unclustered data this will be the input to the function to actually cluster them
errorList = [] #List of all error in the same order as the previous list.
location = []

for set in inputFile:
    location.append(set[0])
    for track in set[1]:
        trackList.append(track[0])
        errorList.append(track[1])
numOfClusters = len(location) #This is the variable for how many clusters will be calculated. This can either be an input or decided from

#===============================================SETUP========================================================#

print("Started embedding...")
sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
embedding = dwave.embedding.pegasus.find_clique_embedding((len(trackList) * numOfClusters), target_graph=hardware_graph) #Creates a dictionary with the key as the logical Qubit ID and a list of the physical qubit IDs that belong to the logical qubit
#print(embedding)

A_matrix = sampler.adjacency #This is the adjacency matrix

#offset_table = DataFrame(sampler.properties['anneal_offset_ranges'], columns=['Min Ofset', 'Max Offset'])
#offset_table.head()
#offset_min = offset_table['Min Offset']
#offset_max = offset_table['Max Offset']

#Lets make ^this^ into either a numpy matrix or just nested lists
#Basically parsing this giant dictionary into just lists
offset_table = sampler.properties['anneal_offset_ranges'] #I think this should return a dictionary with 2 keys and 2 lists
offset_max = offset_table[1] #anneal_offset_ranges is the key in the dict. The values is a list of lists
offset_min = offset_table[0]
v = min(offset_max) #This is the minimum of the maximum offsets. this is possibly a good starting point for what to offset something by. We may need to truncate this later

#===============================================METHODS============================================================#

Q = {}

def calcDist(input1, input2):
    distance = abs(trackList[input1] - trackList[input2])
    totalError = math.sqrt((errorList[input1])**2 + (errorList[input2])**2)
    return distance / totalError

def makeQUBO(points, howManyClusters, strengthOfConstraint):
    numberOfQubits = 0 #Just a counter so we can assign all the keys of the dict Q to the corresponding qubit number
 #This set of loops represents the first term of sigmas in the Hamiltonian
    j = 0
    while j < len(points):
        i = j + 1 #This is so we dont get a distance of 0 because if i=j then we get i-i which = 0
        while i < len(points):    
            k = 0
            while k < howManyClusters:
                x = (k*(len(points))+i)
                y = (k*(len(points))+j)
                Q[(y,x)] = calcDist(i, j)
                k += 1
            i += 1
        j += 1
 #Adjusts all the tracks by a max distance and adds a cosine. This is so that a coupling can be distinguised from noise in the annealer
    maxDist = max(Q.values())
    for i in Q.keys():
        Q[i] = -math.cos((Q[i] * math.pi) / maxDist)
 #This set of loops represents the second term of sigmas in the Hamiltonian
    j = 0
    while j < len(points):
        i = j + 1 #This is so we dont get a distance of 0 because if i=j then we get i-i which = 0
        while i < len(points): 
            x_0 = (0*(len(points))+i) #This is x and y for when k,m = 0
            y_0 = (0*(len(points))+j)
            x_1 = (1*(len(points))+i) #This is x and y for when k,m = 1
            y_1 = (1*(len(points))+j)
         #Example cpupling that this if-statement adds: i=0, j=1, k=0, m=1
             #This checks if there is already an interaction in the QUBO. If so it adds to it instead of overriding it.
            Q[(y_0,x_1)] = math.tanh(1/calcDist(i, j))
         #Example cpupling that this if-statement adds: i=0, j=1, k=1, m=0
            Q[(y_1,x_0)] = math.tanh(1/calcDist(i, j))
            i += 1
        j += 1
    
 #This is adding the biases to make sure each point only picks one cluster. It is the thrid group of sigmas in the Hamiltonian
    i = 0
    while i < len(points):
        k = 0
        while k < howManyClusters:
            x = (k*(len(points))+i)
            Q[(x,x)] = -(strengthOfConstraint)
            k += 1
        m = 0
        l = m + 1
        x = (m*(len(points))+i)
        y = (l*(len(points))+i)
        while m < howManyClusters:
            while l < howManyClusters:
                Q[(x,y)] = 2*(strengthOfConstraint) 
                l += 1
            m += 1
        i += 1

def cluster(reads, offsetList):
    target_Q = dwave.embedding.embed_qubo(Q, embedding, sampler.adjacency, chain_strength=1.0)
    embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw", anneal_offsets=offsetList) #Check that the offsets is the right parameter
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    response = dwave.embedding.unembed_sampleset(embedded_response, embedding, bqm)
    return response.record.sample

def tabulate(data, printBool):
    tabulatedData = {str: int}
    for i in data:
        if str(i) in tabulatedData.keys():
            tabulatedData[str(i)] += 1
        else:
            tabulatedData[str(i)] = 1
        #print(i)
    print(tabulatedData)
    if printBool:
        print(str(data))

def createOffsets():
    offsets = [] #Declaration for now. Edit stuff below this point
    offsets = [0] * len(sampler.properties['anneal_offset_ranges'])
    return offsets

def printQubitGraph(): #Maybe make the name of the file saved an input?
    #name = str(numOfClusters) + "V" + str(len(trackList)) + "T_" + computer
    plt.figure(figsize=(40,40))
    dnx.draw_pegasus_embedding(hardware_graph, emb=embedding, node_size=100, width=2, unused_color=(0,0,0,.25))
    plt.savefig(str(numOfClusters) + "V" + str(len(trackList)) + "T_" + computer + ".png") #That input would go here
    plt.close()

#===============================================FINAL FUNCTION CALLS==========================================#

print("Started sinal solve...")
makeQUBO(trackList, numOfClusters, 1) #This generates a Hamiltonian with the error adjusted points, the amount of cluster specified at the beginning, and with a stregth of constraint of 1
tabulate(cluster(10, createOffsets()), False) #This tabulates the results from the QPU with 10 reads. It does not print the raw output from the QPU. It also takes the offsets that were created by the createOffsets function
printQubitGraph() #Uncomment this if you want the code to save the image.