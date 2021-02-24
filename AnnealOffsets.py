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
#import numpy as numpy #it's been a long day 
import numpy as np
from scipy.sparse import coo_matrix
#from helpers import find_multiple_embeddings
import time
import os

token = "put your token here :)"
computer = "Advantage_system3.1"

#=================================================SETUP=========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
A_matrix = sampler.adjacency #This is the adjacency matrix

#================================================JSON READER=====================================================#

print("Started reading json file...")
#2V40T inputFile = [[0.619949839686173, [[0.6201755054922214, 0.0004166666666666667], [0.6192006233972701, 0.0005833333333333333], [0.6207376504040393, 0.0004166666666666667], [0.6190406640308416, 0.0009166666666666668], [0.6202956081170473, 0.0004166666666666667], [0.6204226895306945, 0.0009166666666666668], [0.619552084928523, 0.0003333333333333334], [0.620112858971073, 0.0004166666666666667], [0.6196292976372996, 0.0013833333333333334], [0.621064412590791, 0.00205], [0.6209727807884767, 0.0013833333333333334], [0.6202057685272748, 0.0009166666666666668], [0.6205460316941176, 0.0005833333333333333], [0.6200698907957599, 0.0004166666666666667], [0.6184321606871547, 0.0013833333333333334], [0.6196795986094037, 0.0004166666666666667], [0.6198699251659776, 0.0013833333333333334], [0.6198822289009036, 0.0005833333333333333], [0.6195363137659851, 0.0004166666666666667], [0.620667676692613, 0.0004166666666666667]]]                        , [0.7644644052662664, [[0.7648474096270056, 0.0005833333333333333], [0.764629660488086, 0.0003333333333333334], [0.7634440363682118, 0.0004166666666666667], [0.7654512098472598, 0.0013833333333333334], [0.764528127739269, 0.0004166666666666667], [0.7642966788475329, 0.0013833333333333334], [0.764755000398421, 0.0005833333333333333], [0.7652707083245642, 0.0005833333333333333], [0.7644544000797732, 0.0004166666666666667], [0.7640091833055777, 0.0013833333333333334], [0.7648443061117294, 0.0004166666666666667], [0.7654803235039588, 0.0005833333333333333], [0.7644352157389597, 0.0003333333333333334], [0.7626683541781973, 0.0009166666666666668], [0.7639046361035419, 0.0009166666666666668], [0.7647682664914893, 0.0009166666666666668], [0.7633141963245973, 0.0009166666666666668], [0.7651315503772118, 0.0004166666666666667], [0.7654841850315482, 0.0009166666666666668], [0.7653116213608953, 0.0009166666666666668]]]]
#inputFile = [[0.619949839686173, [[0.6201755054922214, 0.0004166666666666667], [0.6192006233972701, 0.0005833333333333333], [0.6207376504040393, 0.0004166666666666667], [0.6190406640308416, 0.0009166666666666668], [0.6202956081170473, 0.0004166666666666667], [0.6204226895306945, 0.0009166666666666668], [0.619552084928523, 0.0003333333333333334], [0.620112858971073, 0.0004166666666666667], [0.6196292976372996, 0.0013833333333333334], [0.621064412590791, 0.00205], [0.6209727807884767, 0.0013833333333333334], [0.6202057685272748, 0.0009166666666666668], [0.6205460316941176, 0.0005833333333333333], [0.6200698907957599, 0.0004166666666666667], [0.6184321606871547, 0.0013833333333333334], [0.6196795986094037, 0.0004166666666666667], [0.6198699251659776, 0.0013833333333333334], [0.6198822289009036, 0.0005833333333333333], [0.6195363137659851, 0.0004166666666666667], [0.620667676692613, 0.0004166666666666667]]]                        , [0.7644644052662664, [[0.7648474096270056, 0.0005833333333333333], [0.764629660488086, 0.0003333333333333334], [0.7634440363682118, 0.0004166666666666667], [0.7654512098472598, 0.0013833333333333334], [0.764528127739269, 0.0004166666666666667], [0.7642966788475329, 0.0013833333333333334], [0.764755000398421, 0.0005833333333333333], [0.7652707083245642, 0.0005833333333333333], [0.7644544000797732, 0.0004166666666666667], [0.7640091833055777, 0.0013833333333333334], [0.7648443061117294, 0.0004166666666666667], [0.7654803235039588, 0.0005833333333333333], [0.7644352157389597, 0.0003333333333333334], [0.7626683541781973, 0.0009166666666666668], [0.7639046361035419, 0.0009166666666666668], [0.7647682664914893, 0.0009166666666666668], [0.7633141963245973, 0.0009166666666666668], [0.7651315503772118, 0.0004166666666666667], [0.7654841850315482, 0.0009166666666666668], [0.7653116213608953, 0.0009166666666666668]]]]
inputFile = [[0.5230456769168634, [[0.5240341355642414, 0.0009166666666666668], [0.5235322589723301, 0.0004166666666666667]]], [0.4017284489848494, [[0.4023188409447569, 0.0004166666666666667], [0.4015596160807172, 0.0009166666666666668]]], [0.46472244197733575, [[0.4650397642828156, 0.0004166666666666667], [0.46458608339970275, 0.0013833333333333334]]]]

trackList = [] #This is just a big list of the unclustered data this will be the input to the function to actually cluster them
errorList = [] #List of all error in the same order as the previous list.
location = []

for set in inputFile:
    location.append(set[0])
    for track in set[1]:
        trackList.append(track[0])
        errorList.append(track[1])
numOfClusters = len(location) #This is the variable for how many clusters will be calculated. This can either be an input or decided from
probTitle = str(numOfClusters) + "V" + str(len(trackList)) + "T"
embedding_filename = probTitle + "_" + computer
timeEstimate = (1.5992 * (len(trackList) * numOfClusters)**2 + 12.282 * (len(trackList) * numOfClusters) + 13.817) / 60

#===============================================METHODS============================================================#

Q = {}

def calcDist(input1, input2):
    distance = abs(trackList[input1] - trackList[input2])
    totalError = math.sqrt((errorList[input1])**2 + (errorList[input2])**2)
    return distance / totalError

def makeQUBO(points, howManyClusters, strengthOfConstraint):
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
        """
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
        """

        while i < len(points):
            k = 0
            while k < howManyClusters:
                m = k + 1
                while m < howManyClusters:
                    x_m = (m*(len(points))+i) 
                    y_m = (m*(len(points))+j)
                    x_k = (k*(len(points))+i) 
                    y_k = (k*(len(points))+j)
                    Q[(y_m,x_k)] = math.tanh(1/calcDist(i, j))
                    Q[(y_k,x_m)] = math.tanh(1/calcDist(i, j))
                    m += 1
                k += 1
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

def cluster(reads, offsetList, useOffsets):
    target_Q = dwave.embedding.embed_qubo(Q, embedding, sampler.adjacency, chain_strength=1.0)
    if useOffsets == False: #Easy way to use offsets or not based on an input to the function
        print("Clustering without offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw")
    else:
        print("Clustering with offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw", anneal_offsets=offsetList) 
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
    print(tabulatedData)
    if printBool:
        print(str(data))

def createOffsets():
    offsets = [] #Declaration for now. Edit stuff below this point
    offsets = [0] * len(sampler.properties['anneal_offset_ranges'])
    """
    logical_J = coo_matrix(
        (-np.ones(u.shape[0]), (u, v)),
        shape=(3, 3)
        )
    logical_h = np.array([[0, 0.9, -1]]).T
    sols = np.array(dw2x_output_baseline['solutions']).T
    sols = reshape_solutions(sols, embeddings)
    dist = calculate_solution_distribution(sols)
    delta = -(sols.T * (logical_J + logical_J.T)).T * sols 
    delta = delta - logical_h * sols
    mean_effective_field = np.mean(np.abs(delta.T), axis=0)
    norm_mef = mean_effective_field / np.max(mean_effective_field)  # normalize
    """
    return offsets

def printQubitGraph(): 
    #name = str(numOfClusters) + "V" + str(len(trackList)) + "T_" + computer
    plt.figure(figsize=(40,40))
    dnx.draw_pegasus_embedding(hardware_graph, emb=embedding, node_size=100, width=2, unused_color=(0,0,0,.25))
    plt.savefig(embedding_filename + ".png") 
    plt.close()

def checkembedding():
    print("Started embedding...")
    if os.path.exists((embedding_filename + ".json")):
        print("Found existing embedding!")
        embedding_file = open((embedding_filename + ".json"))
        embedding = json.load(embedding_file)
        embedding = {int(k): v for k, v in embedding.items()}
    else:
        print(f'A {probTitle} problem should take about {timeEstimate:.2f} minutes to embed')
        start = time.process_time()
        embedding = dwave.embedding.pegasus.find_clique_embedding((len(trackList) * numOfClusters), target_graph=hardware_graph)
        outfile = open((embedding_filename + ".json"), "w")
        json.dump(embedding, outfile)
        outfile.close() 
    return embedding

#===============================================EMBEDDING========================================================#

#offset_table = DataFrame(sampler.properties['anneal_offset_ranges'], columns=['Min Ofset', 'Max Offset'])
#offset_table.head()
#offset_min = offset_table['Min Offset']
#offset_max = offset_table['Max Offset']

#Basically parsing this giant dictionary into just lists
offset_table = sampler.properties['anneal_offset_ranges'] #I think this should return a dictionary with 2 keys and 2 lists
offset_max = offset_table[1] #anneal_offset_ranges is the key in the dict. The values is a list of lists
offset_min = offset_table[0]
v = min(offset_max) #This is the minimum of the maximum offsets. this is possibly a good starting point for what to offset something by. We may need to truncate this later

#===============================================FINAL FUNCTION CALLS==========================================#

embedding = checkembedding()
print("Started final solve...")
makeQUBO(trackList, numOfClusters, 1) #This generates a Hamiltonian with the error adjusted points, the amount of cluster specified at the beginning, and with a stregth of constraint of 1

#This tabulates the results from the QPU with a specified number of reads. It also takes the offsets that were created by the createOffsets function. 
#The first bool is for if the offsets should be used or not. The second is for if it should print the raw output from the QPU (untabulated)
tabulate(cluster(1, createOffsets(), False), False) 
#printQubitGraph() #Uncomment this if you want the code to save the image.