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
token = "DEV-176c67d38a18f8e993b4bd11b99aae4d68ab85a3"
computer = "Advantage_system1.1"

#=================================================SETUP=========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
A_matrix = sampler.adjacency #This is the adjacency matrix

#================================================JSON READER=====================================================#

print("Started reading json file...")
#2V40T
#inputFile = [[0.619949839686173, [[0.6201755054922214, 0.0004166666666666667], [0.6192006233972701, 0.0005833333333333333], [0.6207376504040393, 0.0004166666666666667], [0.6190406640308416, 0.0009166666666666668], [0.6202956081170473, 0.0004166666666666667], [0.6204226895306945, 0.0009166666666666668], [0.619552084928523, 0.0003333333333333334], [0.620112858971073, 0.0004166666666666667], [0.6196292976372996, 0.0013833333333333334], [0.621064412590791, 0.00205], [0.6209727807884767, 0.0013833333333333334], [0.6202057685272748, 0.0009166666666666668], [0.6205460316941176, 0.0005833333333333333], [0.6200698907957599, 0.0004166666666666667], [0.6184321606871547, 0.0013833333333333334], [0.6196795986094037, 0.0004166666666666667], [0.6198699251659776, 0.0013833333333333334], [0.6198822289009036, 0.0005833333333333333], [0.6195363137659851, 0.0004166666666666667], [0.620667676692613, 0.0004166666666666667]]]                        , [0.7644644052662664, [[0.7648474096270056, 0.0005833333333333333], [0.764629660488086, 0.0003333333333333334], [0.7634440363682118, 0.0004166666666666667], [0.7654512098472598, 0.0013833333333333334], [0.764528127739269, 0.0004166666666666667], [0.7642966788475329, 0.0013833333333333334], [0.764755000398421, 0.0005833333333333333], [0.7652707083245642, 0.0005833333333333333], [0.7644544000797732, 0.0004166666666666667], [0.7640091833055777, 0.0013833333333333334], [0.7648443061117294, 0.0004166666666666667], [0.7654803235039588, 0.0005833333333333333], [0.7644352157389597, 0.0003333333333333334], [0.7626683541781973, 0.0009166666666666668], [0.7639046361035419, 0.0009166666666666668], [0.7647682664914893, 0.0009166666666666668], [0.7633141963245973, 0.0009166666666666668], [0.7651315503772118, 0.0004166666666666667], [0.7654841850315482, 0.0009166666666666668], [0.7653116213608953, 0.0009166666666666668]]]]
#3V9T
#inputFile = [[0.5230456769168634, [[0.5240341355642414, 0.0009166666666666668], [0.5235322589723301, 0.0004166666666666667]]], [0.4017284489848494, [[0.4023188409447569, 0.0004166666666666667], [0.4015596160807172, 0.0009166666666666668]]], [0.46472244197733575, [[0.4650397642828156, 0.0004166666666666667], [0.46458608339970275, 0.0013833333333333334]]]]
#2V4T
inputFile = [[0.3576734259547776, [[0.3578421714920063, 0.0003333333333333334], [0.35763199025343234, 0.0003333333333333334]]], [0.43751589332926805, [[0.4378207965666029, 0.0005833333333333333], [0.4375708301205863, 0.0004166666666666667]]]]
#3V60T - Does not work!
#inputFile = [[0.6182887356635892, [[0.618367667063972, 0.0004166666666666667], [0.6184721973989786, 0.0004166666666666667], [0.6180590218713224, 0.0005833333333333333], [0.6187684496386381, 0.0009166666666666668], [0.6151634653745182, 0.0013833333333333334], [0.6177003623418004, 0.0003333333333333334], [0.6183292056398593, 0.0005833333333333333], [0.6179900395909106, 0.0004166666666666667], [0.6187060761417905, 0.0005833333333333333], [0.6185141959692401, 0.0004166666666666667], [0.6189844269938084, 0.0003333333333333334], [0.6203748976997095, 0.0013833333333333334], [0.617496285361473, 0.0003333333333333334], [0.619940822516279, 0.0013833333333333334], [0.6183204966575726, 0.0004166666666666667], [0.6169431578680705, 0.00205], [0.6182427636442165, 0.0004166666666666667], [0.6182818637901073, 0.0005833333333333333], [0.6188254617156409, 0.0003333333333333334], [0.6168475339902244, 0.0009166666666666668]]], [0.3498433713498888, [[0.34986555849962686, 0.0003333333333333334], [0.3471051107695432, 0.0013833333333333334], [0.35076060111718194, 0.0004166666666666667], [0.35016724599378884, 0.0013833333333333334], [0.3490452675163178, 0.0013833333333333334], [0.35291780820175567, 0.00205], [0.3500937266410288, 0.0003333333333333334], [0.3500871687513101, 0.0003333333333333334], [0.347070105963216, 0.00205], [0.34955084239764583, 0.0004166666666666667], [0.3525081661050469, 0.00205], [0.3511367437159811, 0.0009166666666666668], [0.3486201717056857, 0.00205], [0.3483867254890516, 0.00205], [0.3487526792901508, 0.0009166666666666668], [0.34967244150068577, 0.0004166666666666667], [0.34831174474431564, 0.0005833333333333333], [0.349640241921386, 0.0005833333333333333], [0.3491922555131508, 0.0005833333333333333], [0.34962715134246736, 0.0004166666666666667]]], [0.4457375279657484, [[0.4455674654265718, 0.0009166666666666668], [0.44601811793302054, 0.0013833333333333334], [0.44525921616303316, 0.0004166666666666667], [0.4428307100457835, 0.00205], [0.44717355159020666, 0.0009166666666666668], [0.4465057414471222, 0.0004166666666666667], [0.44500631294576065, 0.0009166666666666668], [0.44593011559272167, 0.0003333333333333334], [0.4460179699540277, 0.0013833333333333334], [0.44448546108389286, 0.0009166666666666668], [0.4453421872818261, 0.0003333333333333334], [0.4445704555424429, 0.0009166666666666668], [0.44608349228734334, 0.0004166666666666667], [0.4451616867161537, 0.0009166666666666668], [0.4441740001465051, 0.0009166666666666668], [0.44657450963423695, 0.0013833333333333334], [0.4457635133679903, 0.0004166666666666667], [0.4464590878422914, 0.0005833333333333333], [0.44489785002700954, 0.0004166666666666667], [0.44697328275957077, 0.0013833333333333334]]]]
#3V45T
#inputFile = [[0.5824463491006233, [[0.5819963500254111, 0.0009166666666666668], [0.5822157902157752, 0.0004166666666666667], [0.5829594961624105, 0.00205], [0.5832932208997521, 0.0005833333333333333], [0.5825170621551246, 0.0009166666666666668], [0.5820344266381734, 0.0004166666666666667], [0.5822870340375619, 0.0003333333333333334], [0.5802249143090104, 0.0013833333333333334], [0.5821303762695303, 0.0004166666666666667], [0.5825579655207221, 0.0005833333333333333], [0.5814845085684456, 0.0009166666666666668], [0.5860272344844311, 0.00205], [0.5821311862830691, 0.0003333333333333334], [0.5826854502284025, 0.0003333333333333334], [0.5836476451202606, 0.0005833333333333333]]], [0.5434319741656447, [[0.5468784280698337, 0.00205], [0.5425432304748673, 0.0013833333333333334], [0.5420588228398797, 0.0009166666666666668], [0.5455482993419196, 0.00205], [0.5441567848849336, 0.0004166666666666667], [0.543149090037582, 0.0004166666666666667], [0.5434348609039465, 0.0005833333333333333], [0.5436345490249778, 0.0003333333333333334], [0.543548459271061, 0.0009166666666666668], [0.5440218973802, 0.0013833333333333334], [0.54449574522726, 0.0013833333333333334], [0.5433017144780665, 0.0004166666666666667], [0.5426795557890204, 0.0005833333333333333], [0.5411188376076143, 0.00205], [0.5434194625104426, 0.0005833333333333333]]], [0.44557957691390604, [[0.44584167091981164, 0.0005833333333333333], [0.44709137432091356, 0.00205], [0.44618972893141867, 0.0003333333333333334], [0.4454549811187433, 0.0004166666666666667], [0.444759375351496, 0.0004166666666666667], [0.446423439445348, 0.0013833333333333334], [0.4457638873419784, 0.0013833333333333334], [0.44563524639204033, 0.0003333333333333334], [0.4456961080300079, 0.0005833333333333333], [0.4453559741370107, 0.00205], [0.4453152829311786, 0.0004166666666666667], [0.4451449585962526, 0.0013833333333333334], [0.447331214627044, 0.0013833333333333334], [0.44576188893370355, 0.0005833333333333333], [0.4448955126276902, 0.0004166666666666667]]]]
#9V18T
#inputFile = [[0.5809424182323185, [[0.5798524041366876, 0.0005833333333333333], [0.5803906550459442, 0.0013833333333333334]]], [0.5338210669143737, [[0.5334958359143093, 0.0003333333333333334], [0.5353847510632155, 0.00205]]], [0.5872433546695764, [[0.5868250432270549, 0.0013833333333333334], [0.5873695454588539, 0.0003333333333333334]]], [0.28392553267074255, [[0.28468365335623885, 0.0013833333333333334], [0.28433302072608996, 0.00205]]], [0.5273102987865493, [[0.5274207858733031, 0.0013833333333333334], [0.5271072588332316, 0.0009166666666666668]]], [0.5441853638193057, [[0.5441476455449363, 0.0003333333333333334], [0.5438745396561384, 0.0003333333333333334]]], [0.47675466926521926, [[0.47719493131300145, 0.0004166666666666667], [0.4776272096142631, 0.0005833333333333333]]], [0.49608288893572144, [[0.49638798142509954, 0.0003333333333333334], [0.49468076263254696, 0.0013833333333333334]]], [0.44164894665208415, [[0.4406573291803414, 0.0009166666666666668], [0.4419530657371222, 0.0005833333333333333]]]]

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
    print("Finished loop set 1")
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
    print("Finished loop set 2")
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
    print(f"It took {time.process_time() - start:.2f} seconds to make the QUBO")

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
    outfile = open((embedding_filename + "_sols.json"), "w")
    json.dump(embedding, outfile)
    outfile.close()
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
print("Making QUBO...")
makeQUBO(trackList, numOfClusters, 1) #This generates a Hamiltonian with the error adjusted points, the amount of cluster specified at the beginning, and with a stregth of constraint of 1
print("Started final solve...")
#This tabulates the results from the QPU with a specified number of reads. It also takes the offsets that were created by the createOffsets function. 
#The first bool is for if the offsets should be used or not. The second is for if it should print the raw output from the QPU (untabulated)
tabulate(cluster(10, createOffsets(), False), False) 
printQubitGraph() #Uncomment this if you want the code to save the image.