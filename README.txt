SOURCE CODE FOR THE PAPER
*"Universal Topological Regularities of Syntactic Structures: 
Decoupling Efficiency from Optimization"*

Author: Fermín Moscoso Del Prado Martín


1. Requirements:
----------------

The following Python packages are required:
- pyconll
- glob
- pandas
- dfply
- seaborn
- matplotlib
- numpy
- scipy
- plotnine
- networkx

In addition, the code assumes that a local copy of Universal Dependencies v.2.11 exists in the environment, with a directory structure:
			Universal Dependencies 2.11/ud-treebanks-v2.11/

The location of the Universal Dependencies base directory is set in the PATHA variable in runnel.py


2. Contents:
------------

- File "mygraph.py": Contains function to perform all main operations described in the Materials and Methods section, including:
	* EntropyKS: Kolmogorov-Sinai entropy of a graph
	* DegreeDist: Out-degree distribution of a graph
	* RamdomDiTree: Uniform random sampling of directed trees
	* BiasedTree: Non-linear preferential attachment sampling of trees (algorithm S2)
	* MaxHdeg: Maximum value of the out-degree entropy (algorithm S1)
	* COLL2DG: Converts CoNLL format to NetworkX dependency graphs, applying the filters 
		   described in the Materials and Methods
	* MyKLD: 2D gaussian approximation of the Kullback-Leibler divergence.
	* OptimizeL: Apply the optimisation algorithm to a list of graphs
	* Optimize: Apply an optimisation step on one graph
	* RandomModify: Sample an "extended Prüfer neighbour" for a directed tree

- File "runall.py": Performs all the computations to obtain the results in the paper. Running it will 
produce the following files:
	* "langdata.csv": Contains all relevant measures for a sample of graphs built as described 
          in the paper.
	* "trajdata.csv": Average trajectories in optimisation
	* "Fig2.png", "Fig3A.png", "Fig3B.png", "Fig4.png", "FigS1.png", "FigS2.png", and 	
	  "FigS3.png" (each having a .pdf equivalent), are replications of the corresponding figures
	  in the paper

3. Licence (MIT Licence)

Copyright (c) 2023 Fermín Moscoso del Prado Martín

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.