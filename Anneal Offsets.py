#=================================================IMPORTS=========================================================#
import dwave
import dimod
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from dwave.system.samplers import DWaveSampler
from dwave.embedding.pegasus import find_clique_embedding
from pandas import DataFrame
import numpy as numpy #it's been a long day 

token = "put your token here :)"
computer = "Advantage_system1.1"

#===============================================SETUP========================================================#

sampler = DWaveSampler(token=token, solver=computer)
hardware_graph = dnx.pegasus_graph(16, node_list=sampler.properties['qubits'], edge_list=sampler.edgelist)
embedding = dwave.embedding.pegasus.find_clique_embedding(4, target_graph=hardware_graph) #Creates a dictionary with the key as the logical Qubit ID and a list of the physical qubit IDs that belong to the logical qubit
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

def makeQUBO():
    hi = "hi" #Just a placeholder for now

def cluster(reads, offsetList):
    target_Q = dwave.embedding.embed_qubo(Q, embedding, sampler.adjacency, chain_strength=1.0)
    embedded_response = sampler.sample_qubo(target_Q, num_reads=reads, answer_mode="raw", offsets=offsetList)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    response = dwave.embedding.unembed_sampleset(embedded_response, embedding, bqm)
    return response.record.sample

def printQubitGraph(): #Maybe make the name of the file saved an input?
    plt.figure(figsize=(40,40))
    dnx.draw_pegasus_embedding(hardware_graph, emb=embedding, node_size=100, width=2, unused_color=(0,0,0,.3))
    plt.savefig('2V2T_Advantage.png') #That input would go here
    plt.close()

#printQubitGraph() #Uncomment this if you want the code to save the image.