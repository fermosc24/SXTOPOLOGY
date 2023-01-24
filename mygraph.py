#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
from itertools import combinations
from random import sample,choice
import re


def Adjacency(graph,digraph=False):
    """
    Computes adjacency matrix for a graph. To be used instead of NetworkX's
    native function.

    :graph: NetworkX Graph object
    :digraph: Whether to treat the graph as a directed graph.
    :return: Adjacency matrix (NumPy array)
    """ 
    N = len(graph.nodes)
    adj = np.zeros((N,N))
    edges = graph.edges
    for a,b in edges:
        adj[a,b] = 1
        if not digraph:
            adj[b,a] = 1
    return adj

def EntropyKS(graph):
    """
    Computes Komogorov-Sinai entropy of a graph.

    :graph: NetworkX Graph object
    :return: KS-entropy of the graph, in base 2
    """ 
    if len(graph.nodes)>1:
        M = Adjacency(graph)
        try:
            eig = np.real(np.linalg.eig(M)[0])
        except:
            eig = np.ones((1,))
        lambd = np.max(eig)
    else:
        lambd = 1
    return np.log2(np.round(lambd,8))

def _DirectedTree(root,skeleton):
    """
    Auxiliary function for generating a directed tree given its root node
    and skeleton.
    
    :root: Index (from 0 to N-1) of the vertex that should be root
    :graph: NetworkX Graph object
    :return: Digraph (NetworkX DiGraph)
    """ 
    nodel = set([root])
    edges = set(skeleton.edges)
    newEdges = []
    while edges:
        seen = set([])
        ledges = list(edges)
        for a,b in ledges:
            A = a in nodel
            B = b in nodel
            if A and B:
                seen.add((a,b))                
            elif A:
                newEdges.append((a,b))
                nodel.add(b)
                seen.add((a,b))
            elif B:
                newEdges.append((b,a))
                nodel.add(a)
                seen.add((a,b))
        edges -= seen
    return nx.DiGraph(newEdges)              

def RandomDiTree(N):
    """
    Uniformly sampled directed tree with N nodes.

    :N: Number of desired nodes
    :return: NetworkX DiGraph with N nodes
    """ 
    graph = nx.random_tree(N)
    root = int(np.random.choice(graph.nodes,1)[0])
    return _DirectedTree(root,graph)

def PruferDiTree(prufer):
    """
    Reconstruct directed tree from an extended Prüfer sequence.

    :prufer: Extended Prüfer sequence, whose first element is the root identity
    :return: NetworkX DiGraph corresponding to the Prüfer sequence
    """ 
    graph = nx.from_prufer_sequence(prufer[1:])
    root = prufer[0]
    return _DirectedTree(root,graph)

def DegreeDist(graph,maxd=30,directed=False):
    """
    Computes degree distribution of a graph

    :graph: NetworkX Graph
    :maxd: Maximum number for the degree
    :directed: If True, compute out_degree distribution
    :return: Array with node degree frequencies (index is the degree)
    """ 
    dicth = np.zeros((maxd))
    if directed:
        degree_sequence = sorted((d for n, d in graph.out_degree()),
                                 reverse=True)
    else:
        degree_sequence = sorted((d for n, d in graph.degree()),
                                 reverse=True)
    degrees,counts = np.unique(degree_sequence, return_counts=True)
    dicth[list(degrees)]=counts
    return dicth
    
def EntropyD(dist):
    """
    Computes Shannon entropy of a histograph

    :dist: Histograph (array of node degree frequencies)
    :return: Shannon entropy on base 2
    """ 
    if np.any(dist>0):
        dist2 = dist/np.sum(dist)
        return -np.sum(dist2[dist2>0]*np.log2(dist2[dist2>0]))
    else:
        return 0
    
def RandomModify(digraph):
    """
    Randomly sample an extended Prüfer neighbor from a directed tree

    :digraph: NetworkX DiGraph 
    :return: NetworkX DiGraph
    """ 
    nodes = set(digraph.nodes)
    N = len(nodes)
    adj = Adjacency(digraph,digraph=True)
    root = int(np.where(np.all(adj == 0,0))[0])      
    prufer = nx.to_prufer_sequence(nx.Graph(digraph))
    prufer = [root]+prufer
    i = np.random.choice(list(range(N-1)))
    old = prufer[i]
    prufer[i] = np.random.choice(list(nodes-{old}))  
    return PruferDiTree(prufer)
    
def OptimizeL(L,nepochs=10,rho=.5,noise=.05,npred=200,Href=None):
    """
    Optimizes a list of trees for nepochs, comparing to a reference distribution,
    when provided

    :t: List of trees (NetworkX DiGraphs)
    :nepochs: Number of generations for optimization
    :rho: Optimization paramater rho
    :noise: Optimization parameter sigma
    :npred: maximum K size to consider
    :Href: If provided, reference distribution against which to compare
    :return: Tuple (- List of optimized trees
                    - Array with H_deg and H_ks values for each tree in each
                      optimization generation (N_trees x 2 x nepochs)
                    - Array with number of accepted changes by epoch
                    - Array with estimated KLD between optimized and reference
                      distributions for each epoch)
    """ 
    N = len(L)    
    HS = np.zeros((N,2,nepochs+1))

    Lp = []
    error = []

    for i in range(N):
        t = L[i]
        h = EntropyKS(t)
        dd0 = DegreeDist(t,npred,directed=True)
        s = EntropyD(dd0)
        Lp.append(t)
        HS[i,:,0]=[h,s]
    
    if not Href is None:
        error.append(MyKLD(Href,HS[:,:,0]))

    Changes = []
    for epoch in range(nepochs):
        Lp2 = []
        changes = 0
        for nn,t in enumerate(Lp):
            h0 = HS[nn,0,epoch]
            s0 = HS[nn,1,epoch]

            t1,h1,s1,change = Optimize(t,h0,s0,rho,noise,npred)
            Lp2.append(t1)
            HS[nn,:,epoch+1] = np.array([h1,s1])
            changes+=change
        Lp = Lp2
        Changes.append(changes)
        if not Href is None:
            error.append(MyKLD(Href,HS[:,:,epoch+1]))
    return(Lp,HS,np.array(Changes),np.array(error))
           
def Optimize(t,h0,s0,rho,noise=0,npred=200):
    """
    Performs 1 step of optimization:
        - Generate extended Prüfer neighbor
        - Compute h_ks', h_deg'
        - If new tree is better, return new tree, otherwise return old tree

    :t: Original tree (NetworkX DiGraph)
    :h0: H_deg of t
    :s0: H_ks of t
    :rho: Optimization paramater rho
    :noise: Optimization parameter sigma
    :npred: maximum K size to consider
    :return: Tuple (Chosen tree (NetworkX DiGraph),h_deg,h_ks,binary value
                    indicating if modification took place)
    """ 
    v0 = rho*h0 - (1-rho)*s0

    newt = RandomModify(t)
    h1 = EntropyKS(newt)
    dd1 = DegreeDist(newt,npred,directed=True)
    s1 = EntropyD(dd1)
    v1 = rho*h1 - (1-rho)*s1
    
    if v1+np.random.normal(0,noise)>v0:
        return newt,h1,s1,1
    else:
        return t,h0,s0,0
    
def __value(t,N):
    """
    Auxiliary function for MaxHdeg
    """
    _,hist=np.unique(t, return_counts=True)
    hist=list(hist)
    hist.append(N-np.sum(hist))
    return EntropyD(np.array(hist))

 
def MaxHdeg(N):
    """
    Maximum possible out_degree entropy of a Digraph with N nodes.
    Implements Algorithm S1 in the Supplementary Materials.
    
    :N: Number of nodes (int)
    :return: Maximum out_degree entropy in base 2 (float)
    """
    Seen = set([])
    starter = tuple([1]*(N-1))
    Stack = [starter]
    best = 0
    while Stack:
        t = Stack.pop()
        v = __value(t,N)
        if v>=best:
            best = v
        L = len(t)
        if L>1:
            for i,j in combinations(range(L),2):
                t2 = list(t)
                t2[i] += t[j]
                t2.pop(j)
                t2 = tuple(sorted(t2))
                if not t2 in Seen:
                    Stack.append(t2)
                    Seen.add(t2)
    return best

def MinHdeg(N):
    """
    Minimum possible out_degree entropy of a Digraph with N nodes
   
    :N: Number of nodes (int)
    :return: Minimum out_degree entropy in base 2 (float)
    """    
    p0 = 1/N  # Out degree 0
    p1 = 1-p0 # Out degree 1
    return -p0*np.log2(p0)-p1*np.log2(p1)


def MinHks(N):
    """
    Minimum possible Komogorov-Sinai entropy of a graph with N nodes
   
    :N: Number of nodes (int)
    :return: Minimum K-S entropy in base 2 (float)
    """    
    return EntropyKS(nx.Graph([(i,i+1) for i in range(N-1)]))


def MaxHks(N):
    """
    Maximum possible Komogorov-Sinai entropy of a graph with N nodes
   
    :N: Number of nodes (int)
    :return: Maximum K-S entropy in base 2 (float)
    """    
    return np.log2(N-1)/2

def BiasedTree(N,alpha=0.):
    """
    Returns a random tree sampled by non-linear preferential attachment with
    exponent alpha.
    Implements Algorithm S2 in the Supplementary Materials.

   
    :N: Number of nodes (int)
    :alpha: Nonlinear exponent
    :return: Maximum K-S entropy in base 2 (float)
    """    
    free = sample(range(N),N)
    nodes = [free.pop()]
    links = []
    K = np.zeros((N,))
    K[nodes[0]]=1.
    while free:
        newn = free.pop()
        K[newn]=1.
        p = K[np.array(nodes)]**alpha
        p = p/np.sum(p)
        mother = np.random.choice(nodes,p=p)
        K[mother] += 1.
        nodes.append(newn)
        links.append((mother,newn))
    return nx.DiGraph(links)

def CONLL2DG(sentence,reg="PUNCT"):
    """
    Preprocess a dependency tree in CoNLL format
    - removes nodes with type reg
    - removes range nodes
   
    :sentence: List of triplets (is, token, upos)
    :reg: Pos to exclude
    :return: NetworkX DiGraph
    """    
    nodes = [(token.id,token.head,token.upos)\
                for token in sentence if re.match(r"^[0-9]+$",token.id)]
    i = 0
    convert = {}
    edges0 = []
    for (a,h,t) in nodes:
        if t!=reg:
            convert[a]=i
            if h!=0:
                edges0.append((h,a))
            i+=1
    edges = []
    for (a,b) in edges0:
        if a in convert and b in convert:
            edges.append((convert[a],convert[b]))
    return nx.DiGraph(edges)

def MyKLD(X,Y):
    """
    2D Gaussian approximation of the Kullback-Leibler divergence between two
    2D distributions, assuming they are 2D Gaussian samples
   
    :X: 2D array whose rows are 2-dimensional samples from a distribution
    :Y: 2D array whose rows are 2-dimensional samples from a distribution
    :return: Estimated KLD
    """    
    mu1,mu2 = tuple(np.mean(X,axis=0))
    sigma1,sigma2 = tuple(np.std(X,axis=0))
    m1,m2 = tuple(np.mean(X,axis=0))
    s1,s2 = tuple(np.std(X,axis=0))
    rho = np.corrcoef(X,rowvar=False)[0,1]
    r = np.corrcoef(Y,rowvar=False)[0,1]
    
    return (
    ((mu1-m1)**2/s1**2 - 2*r*(mu1-m1)*(mu2-m2)/(s1*s2) + (mu2-m2)**2/s2**2) /
    (2 * (1 - r**2)) +
    ((sigma1**2-s1**2)/s1**2 - 2*r*(rho*sigma1*sigma2-r*s1*s2)/(s1*s2) + 
     (sigma2**2-s2**2)/s2**2) /
    (2 * (1 - r**2)) +
    np.log((s1**2 * s2**2 * (1-r**2)) / (sigma1**2 * sigma2**2 * (1-rho**2))) / 2
    )
