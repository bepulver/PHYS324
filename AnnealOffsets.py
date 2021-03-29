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
import pandas as pd 
#import numpy as numpy #it's been a long day 
import numpy as np
from scipy.sparse import coo_matrix
#from helpers import find_multiple_embeddings
import time
import os

token = "put your token here :)"
token = "DEV-176c67d38a18f8e993b4bd11b99aae4d68ab85a3"
computer = "Advantage_system1.1"

#=================================================SETUP=========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
A_matrix = sampler.adjacency #This is the adjacency matrix

#================================================JSON READER=====================================================#

#2V40T
#inputFile = [[0.619949839686173, [[0.6201755054922214, 0.0004166666666666667], [0.6192006233972701, 0.0005833333333333333], [0.6207376504040393, 0.0004166666666666667], [0.6190406640308416, 0.0009166666666666668], [0.6202956081170473, 0.0004166666666666667], [0.6204226895306945, 0.0009166666666666668], [0.619552084928523, 0.0003333333333333334], [0.620112858971073, 0.0004166666666666667], [0.6196292976372996, 0.0013833333333333334], [0.621064412590791, 0.00205], [0.6209727807884767, 0.0013833333333333334], [0.6202057685272748, 0.0009166666666666668], [0.6205460316941176, 0.0005833333333333333], [0.6200698907957599, 0.0004166666666666667], [0.6184321606871547, 0.0013833333333333334], [0.6196795986094037, 0.0004166666666666667], [0.6198699251659776, 0.0013833333333333334], [0.6198822289009036, 0.0005833333333333333], [0.6195363137659851, 0.0004166666666666667], [0.620667676692613, 0.0004166666666666667]]]                        , [0.7644644052662664, [[0.7648474096270056, 0.0005833333333333333], [0.764629660488086, 0.0003333333333333334], [0.7634440363682118, 0.0004166666666666667], [0.7654512098472598, 0.0013833333333333334], [0.764528127739269, 0.0004166666666666667], [0.7642966788475329, 0.0013833333333333334], [0.764755000398421, 0.0005833333333333333], [0.7652707083245642, 0.0005833333333333333], [0.7644544000797732, 0.0004166666666666667], [0.7640091833055777, 0.0013833333333333334], [0.7648443061117294, 0.0004166666666666667], [0.7654803235039588, 0.0005833333333333333], [0.7644352157389597, 0.0003333333333333334], [0.7626683541781973, 0.0009166666666666668], [0.7639046361035419, 0.0009166666666666668], [0.7647682664914893, 0.0009166666666666668], [0.7633141963245973, 0.0009166666666666668], [0.7651315503772118, 0.0004166666666666667], [0.7654841850315482, 0.0009166666666666668], [0.7653116213608953, 0.0009166666666666668]]]]
#3V6T
#inputFile = [[0.5230456769168634, [[0.5240341355642414, 0.0009166666666666668], [0.5235322589723301, 0.0004166666666666667]]], [0.4017284489848494, [[0.4023188409447569, 0.0004166666666666667], [0.4015596160807172, 0.0009166666666666668]]], [0.46472244197733575, [[0.4650397642828156, 0.0004166666666666667], [0.46458608339970275, 0.0013833333333333334]]]]
#2V4T
#inputFile = [[0.3576734259547776, [[0.3578421714920063, 0.0003333333333333334], [0.35763199025343234, 0.0003333333333333334]]], [0.43751589332926805, [[0.4378207965666029, 0.0005833333333333333], [0.4375708301205863, 0.0004166666666666667]]]]
#2V50T
#inputFile = [[0.40372712245907655, [[0.40313230894383995, 0.0005833333333333333], [0.40128689560282804, 0.0013833333333333334], [0.40411536912117757, 0.0003333333333333334], [0.40229746198479316, 0.0013833333333333334], [0.4026530582972256, 0.0005833333333333333], [0.40496019263698085, 0.00205], [0.4035448749007843, 0.0005833333333333333], [0.40330605065684577, 0.0009166666666666668], [0.40395058872927236, 0.0009166666666666668], [0.4046022135805178, 0.0005833333333333333], [0.40292238395410696, 0.0009166666666666668], [0.4023013958592916, 0.0009166666666666668], [0.40286435832905343, 0.00205], [0.4046331997053262, 0.0004166666666666667], [0.4037195224233967, 0.0005833333333333333], [0.40289850548321177, 0.0003333333333333334], [0.40256632631597933, 0.00205], [0.40284779126459225, 0.0009166666666666668], [0.4030053912344479, 0.0005833333333333333], [0.4073209909011575, 0.00205], [0.4047261154188145, 0.00205], [0.4039366745709197, 0.0004166666666666667], [0.40355421844393713, 0.0003333333333333334], [0.4041819489326438, 0.0005833333333333333], [0.40248173744565413, 0.0013833333333333334]]], [0.46567486036165345, [[0.4641909456987078, 0.00205], [0.4656847208660851, 0.0004166666666666667], [0.4651644822671265, 0.0005833333333333333], [0.46520789913516536, 0.0003333333333333334], [0.4654909585992463, 0.0004166666666666667], [0.46478660256079696, 0.0009166666666666668], [0.46403649879472936, 0.0013833333333333334], [0.46554608568856276, 0.0003333333333333334], [0.4646012247510719, 0.0004166666666666667], [0.4658324856307101, 0.0003333333333333334], [0.4667948738926254, 0.00205], [0.46429959512639873, 0.00205], [0.465264575691102, 0.0004166666666666667], [0.46611862512493846, 0.0003333333333333334], [0.4653225360875454, 0.0013833333333333334], [0.4661083934494567, 0.0003333333333333334], [0.46444244799491674, 0.0005833333333333333], [0.46617339671279684, 0.0003333333333333334], [0.46583324598545045, 0.0005833333333333333], [0.4653467112466013, 0.0009166666666666668], [0.46659589148806285, 0.00205], [0.4657853869436152, 0.0003333333333333334], [0.4654675477451264, 0.0005833333333333333], [0.46531140580563884, 0.0003333333333333334], [0.46541815969045824, 0.0004166666666666667]]]]
#2V10T
#inputFile = [[0.26459440140957663, [[0.2647385486134077, 0.0003333333333333334], [0.2654162874927705, 0.0005833333333333333], [0.2644212460179555, 0.0004166666666666667], [0.26479940986166234, 0.0005833333333333333], [0.26483538260019035, 0.0003333333333333334]]], [0.6371146815924992, [[0.6370521845916481, 0.0005833333333333333], [0.6371294827443515, 0.0005833333333333333], [0.639912825546594, 0.00205], [0.6365240549681587, 0.0005833333333333333], [0.6375382756189499, 0.0005833333333333333]]]]
#2V16T
inputFile = [[0.3477026272590622, [[0.34670542276621336, 0.0013833333333333334], [0.3479626564874828, 0.0005833333333333333], [0.34719922931666236, 0.0013833333333333334], [0.34582910050787274, 0.0013833333333333334], [0.3484048734552079, 0.0005833333333333333], [0.347763739911817, 0.0005833333333333333], [0.3465146075655664, 0.0009166666666666668], [0.3478775642040726, 0.0013833333333333334]]], [0.5096619032548017, [[0.5093428196876093, 0.0005833333333333333], [0.5095549411829058, 0.0009166666666666668], [0.5102061296299512, 0.0005833333333333333], [0.5087631763603003, 0.0003333333333333334], [0.5086694400220225, 0.0005833333333333333], [0.5098174300893572, 0.0004166666666666667], [0.5094418005132764, 0.0009166666666666668], [0.5093916323048657, 0.0004166666666666667]]]]
#2V20T
#inputFile = [[0.4298720738744714, [[0.42956858524444586, 0.00205], [0.43055210280748696, 0.0005833333333333333], [0.4281751472645296, 0.0013833333333333334], [0.4295458477522827, 0.0004166666666666667], [0.4301522468613366, 0.0009166666666666668], [0.42922161096355904, 0.0005833333333333333], [0.4297410489392506, 0.0004166666666666667], [0.4299412304163768, 0.0004166666666666667], [0.4287570336655881, 0.0013833333333333334], [0.43072908258370507, 0.0009166666666666668]]], [0.5532946625199134, [[0.5536168324913402, 0.0005833333333333333], [0.5533625539869165, 0.0003333333333333334], [0.5528725749640683, 0.0005833333333333333], [0.5535465063402216, 0.0003333333333333334], [0.5535674819591111, 0.0004166666666666667], [0.5535906431076907, 0.0003333333333333334], [0.552889680962357, 0.0004166666666666667], [0.5534116462767563, 0.0013833333333333334], [0.5536775947471168, 0.0003333333333333334], [0.5539587696221367, 0.0004166666666666667]]]]
#2V28T
#inputFile = [[0.31571817267074337, [[0.3156134039706826, 0.0003333333333333334], [0.31455038984443007, 0.00205], [0.3157642770348872, 0.00205], [0.31612599297646893, 0.0003333333333333334], [0.3151971311639035, 0.0004166666666666667], [0.3157155546348742, 0.0004166666666666667], [0.3140892584306134, 0.0013833333333333334], [0.3174205985673443, 0.00205], [0.3161015887926657, 0.0009166666666666668], [0.31716675070661854, 0.0009166666666666668], [0.3155865349881447, 0.0009166666666666668], [0.31493158820318123, 0.0005833333333333333], [0.31719940317050427, 0.0009166666666666668], [0.3170537738756036, 0.0013833333333333334]]], [0.661181258203091, [[0.6596810060112318, 0.0013833333333333334], [0.6614027663673783, 0.0004166666666666667], [0.6612599285720642, 0.0003333333333333334], [0.6616816681316353, 0.0005833333333333333], [0.6614533910848687, 0.0003333333333333334], [0.6613573805812722, 0.0013833333333333334], [0.6616798816899587, 0.0013833333333333334], [0.6594706304599387, 0.00205], [0.6609596580784179, 0.0005833333333333333], [0.661268871267803, 0.0005833333333333333], [0.6611259731926012, 0.0013833333333333334], [0.661821177686621, 0.0003333333333333334], [0.6612670836834323, 0.0003333333333333334], [0.6612472160140108, 0.0004166666666666667]]]]

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

def makeQUBO(points, howManyClusters):
    start = time.process_time()
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
                    #Q[(y_m,x_k)] = math.tanh(1/calcDist(i, j))
                    #Q[(y_k,x_m)] = math.tanh(1/calcDist(i, j))
                    Q[(y_m,x_k)] = math.tanh(1/calcDist(i, j))
                    Q[(y_k,x_m)] = math.tanh(1/calcDist(i, j))
                    m += 1
                k += 1
            i += 1
        j += 1
 #This is adding the biases to make sure each point only picks one cluster. It is the thrid group of sigmas in the Hamiltonian
    print(f"lambda is {Lambda()}")
    i = 0
    while i < len(points):
        k = 0
        while k < howManyClusters:
            x = (k*(len(points))+i)
            Q[(x,x)] = -(Lambda())
            k += 1
        m = 0
        while m < howManyClusters:
            l = m + 1
            while l < howManyClusters:
                x = (m*(len(points))+i)
                y = (l*(len(points))+i)
                Q[(x,y)] = 2*(Lambda())
                l += 1
            m += 1
        i += 1
    print(f"It took {time.process_time() - start:.2f} seconds to make the QUBO")

def cluster(reads, offsetList, useOffsets):
    target_Q = dwave.embedding.embed_qubo(Q, embedding, sampler.adjacency, chain_strength=1.0) #Takes my logical QUBO and turns it into a physical QUBO
    if useOffsets == False: #Easy way to use offsets or not based on an input to the function
        print("Clustering without offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw")
    else:
        print("Clustering with offsets...")
        embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw", anneal_offsets=offsetList) 
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    response = dwave.embedding.unembed_sampleset(embedded_response, embedding, bqm)
    return response.record.sample

def tabulate(data, printBool): #Essentially counts the amount of time that a specific response was returned from the QPU
    tabulatedData = {}
    for i in data:
        if str(i) in tabulatedData.keys():
            tabulatedData[str(i)] += 1
        else:
            tabulatedData[str(i)] = 1
    outfile = open((embedding_filename + "_sols.json"), "w")
    json.dump(tabulatedData, outfile)
    outfile.close()
    print(tabulatedData)
    if printBool:
        print(str(data))
    return tabulatedData

def createOffsets(solutionArray):
    logical_j = [0] * (numOfClusters * len(trackList))
    logical_h = [0] * (numOfClusters * len(trackList))
    offsets = [] 
    #offsets = [0] * len(sampler.properties['anneal_offset_ranges'])
    

    for i in Q.keys(): #Initialize the j list with lists. This essentially is a matrix with MATLAB-like indexing
        logical_j[i[0]] = []
    for i in Q.keys(): #This fills the matrix with the row being the qubit and the column being the coupling between the row number and column number
        if i[0] == i[1]:
            logical_h[i[0]] = Q[i]
        else:
            while (max(i)+1) > len(logical_j[min(i)]):
                logical_j[min(i)].append(0)
            logical_j[min(i)][max(i)] = Q[i]

    avgSpin = [] #This should end up being a list that looks like the solutions but will have fractional pieces of spin corresponding to the average
    for i in range(len(solutionArray[1])):
        numer = 0
        denom = 0
        for j in range(len(solutionArray)):
            numer += solutionArray[j][i]
            denom += 1
        avgSpin.append(numer / denom)
    #finding the "force" on the qubits
    for i in range(len(logical_j)): #iterating through the rows
        for j in range(len(logical_j[i])): #iterating through the columns
            logical_j[i][j] *= avgSpin[j] #Now this is a field (After it's summed)
    for i in range(len(logical_j)):
        logical_j[i] = sum(logical_j[i]) * avgSpin[i] #Now this is a force
    for i in range(len(logical_h)):
        logical_h[i] *= avgSpin[i] #Going from field to force
    
     #Combine the 2 and normalize
    for i in range(len(logical_j)):
        offsets.append(logical_j[i] + logical_h[i])
    maxoffset = max(offsets)
    for i in range(len(offsets)):
        offsets[i] /= maxoffset

    pd.DataFrame(logical_j).to_csv("/workspace/PHYS324/j.csv")
    #print(f"Logical J: {logical_j}\nLogical H: {logical_h}")
    
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
"""
def Lambda():
    
    Dijs = [] 
    oneoverDijs = []

    i = 0
    while i < len(trackList) / 2:
        j = i + 1
        while j < len(trackList) / 2:
            Dijs.append(calcDist(i, j))
            j += 1
        while j < len(trackList):
            oneoverDijs.append(1/calcDist(i, j))
            j += 1
        i += 1
    for i in range(int(len(trackList) / 2), len(trackList)):
        j = i + 1
        while j < len(trackList):
            Dijs.append(calcDist(i, j))
            j += 1
    print(f"Average Dij: {sum(Dijs) / len(Dijs)}")
    print(Dijs)
    print(f"Average 1/Dij: {sum(oneoverDijs) / len(oneoverDijs)}")
    print(oneoverDijs)
    
    choose2 = math.factorial(len(trackList) / numOfClusters) / (2 * math.factorial((len(trackList) / numOfClusters) - 2))
    return ((numOfClusters * choose2) * (sum(Dijs) / len(Dijs)) + (sum(oneoverDijs) / len(oneoverDijs)) * (len(trackList) / numOfClusters) * (len(trackList) - (len(trackList) / numOfClusters)) / (len(trackList) * (1-numOfClusters)**2)) * 1.1
"""

def Lambda():
    return 1

def accuracy(sols, reads): #HARDCODED FOR 2 CLUSTERS AS OF NOW
    ncorrect = 0
    answer1 = str((int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [0])).replace(",", "")
    answer2 = str((int(len(trackList) / 2) * [1]) + (int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [0]) + (int(len(trackList) / 2) * [1])).replace(",", "")
    for i in list(sols):
        if str(i) == answer1:
            ncorrect += 1
        elif str(i) == answer2:
            ncorrect += 1
    e = ncorrect / reads
    delta = 100 * math.sqrt(e * (1 - e) * reads) / reads
    print(f"{e * 100}% were correct with an uncertanty of {delta}")


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

embedding = checkembedding() #either grabs embedding from a json, or calculates it using pegasus
makeQUBO(trackList, numOfClusters) #This generates a Hamiltonian with the error adjusted points, the amount of cluster specified at the beginning, and with a stregth of constraint of 1
print("Started final solve...")

reads = 1000
sols = cluster(reads, [0], False) #This clusters and stores the answers. Uses a placeholder list, but we are using the offsets anyway
#offsets = createOffsets(sols)
#sols = cluster(100, offsets, True)
tabsols = tabulate(sols, False) #Tabulated solutions
accuracy(sols, reads)
#This tabulates the results from the QPU with a specified number of reads. It also takes the offsets that were created by the createOffsets function. 
#The first bool is for if the offsets should be used or not. The second is for if it should print the raw output from the QPU (untabulated)
#tabulate(cluster(1000, createOffsets(), False), False)
printQubitGraph() #Uncomment this if you want the code to save the image.
