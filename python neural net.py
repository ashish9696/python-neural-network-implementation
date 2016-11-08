import numpy as np
import math as m
import random as r
import time as t

def sigmoid(x):
 return 1.0/(1.0+m.exp(-x))
 
vectorizedSigmoid=np.vectorize(sigmoid)
 
def sigmoidDerivative(y): 
 return y*(1-y)  
 
vectorizedSigmoidDerivative=np.vectorize(sigmoidDerivative)

class NeuralNetwork:
	
 def __init__(self,hiddenLayers,samples):
  self.hiddenLayers=hiddenLayers
  self.samples=listsToNumPyMatrixes(samples)
  self.l2Lambda=0.001
  self.alpha=1
  #self.precision=1e-8
  self.convergence=1e-6
  self.converged=False
  self.setLayerSizes()
  self.cost=self.costFunction()
  
 def setLayerSizes(self):
  nInput=len(self.samples[0][0])
  nOutput=len(self.samples[0][1])
  self.nLayers=[nInput]+self.hiddenLayers+[nOutput]
  self.layerSizes=[(self.nLayers[i+1],self.nLayers[i]) for i in range(len(self.nLayers)-1)]
  print('Layer sizes: {}'.format(self.layerSizes))
  self.setInitialWeights()
  
 def setInitialWeights(self):
  self.weights=[]
  for (nRows,nColumns) in self.layerSizes:
   self.weights.append(getRandomMatrix((nRows,nColumns+1)))
   
 def feedForward(self,input):
  activation=input
  for matrix in self.weights:
   activation=addRowToNumPyMatrix(activation,[1])
   activation=vectorizedSigmoid(matrix*activation)
  return activation
   
 def costFunctionPerSample(self,sample):
  (input,output)=sample
  lastA=self.feedForward(input)
  return np.sum(vectorizedLogisticCost(output,lastA))
  
 def costFunction(self):
  costsSum=0
  n=len(self.samples)
  for sample in self.samples:
   costsSum+=self.costFunctionPerSample(sample)
  sss=0
  for matrix in self.weights:
   sss+=np.sum(vectorizedSquare(matrix))
  return -costsSum/n+self.l2Lambda*sss/(2*n)
            
 def costDerivative(self, output_activations, y):
  return (output_activations-y)       
  
 def backpropagation(self,x,y):
  nablaW=[np.zeros(w.shape) for w in self.weights]
  activation=addRowToNumPyMatrix(x,[1])
  activations=[activation]
  zs=[]
  deltas=[]
  for w in self.weights:
   z=w*activation
   zs.append(z)
   activation=addRowToNumPyMatrix(vectorizedSigmoid(z),[1])
   activations.append(activation)
  delta=activations[-1]-addRowToNumPyMatrix(y,[1])
  nablaW[-1]=removeLastRowOfNumPyMatrix(delta*activations[-2].transpose())
  numberOfLayers=len(self.nLayers)
  for l in range(2,numberOfLayers):
   wt=self.weights[-l+1].transpose()
   sd=vectorizedSigmoidDerivative(activations[-l])
   lastDeltaWithoutBias=removeLastRowOfNumPyMatrix(delta)
   delta=np.multiply(wt*lastDeltaWithoutBias,sd)
   nablaW[-l]=removeLastRowOfNumPyMatrix(delta*activations[-l-1].transpose())
  return nablaW

 def backpropagateAllSamples(self):
  nablaWSum=[np.zeros(w.shape) for w in self.weights]
  nMatrix=len(nablaWSum)
  nSamples=len(self.samples)
  for sample in self.samples:
   (x,y)=sample
   nablaW=self.backpropagation(x,y)
   for matrixIndex in range(nMatrix):
   	nablaWSum[matrixIndex]+=nablaW[matrixIndex]
  for matrixIndex in range(nMatrix):
   nablaWSum[matrixIndex]=(nablaWSum[matrixIndex]+self.l2Lambda*self.weights[matrixIndex])/nSamples
  return nablaWSum
  
 def iterate(self):
  g=self.backpropagateAllSamples()
  nMatrix=len(self.weights)
  for matrixIndex in range(nMatrix):
   self.weights[matrixIndex]-=self.alpha*g[matrixIndex]
  totalCost=self.costFunction()
  if(abs(totalCost-self.cost)<self.convergence): self.converged=True
  self.cost=totalCost
  
 def printIteration(self):
  print('Iteration '+str(self.iteration)+'. Error: '+str(self.cost))
  
 def iterateUntilConvergence(self):
  t0=t.time()
  self.iteration=0
  self.printIteration()
  while not self.converged:
   self.iteration+=1
   if self.iteration%100==0: 
   	self.printIteration()
   self.iterate()
  self.printIteration()
  t1=t.time()
  self.deltaTime=t1-t0
  print('Neural network has converged in {:0.2f} seconds.'.format(self.deltaTime))
  
def logisticCost(y,y2): 
 return y*m.log(y2)+(1-y)*m.log(1-y2)
 
vectorizedLogisticCost=np.vectorize(logisticCost)

def square(x):
 return x*x
 
vectorizedSquare=np.vectorize(square)
  
def addRowToNumPyMatrix(m,row):
 return np.vstack([m,row])
 
def removeLastRowOfNumPyMatrix(m):
 return m[:m.shape[0]-1]

def listToNumPyMatrix(l):
 return np.matrix(l).transpose()
 
def columnVectorToList(m):
 return [m[i,0] for i in range(m.shape[0])]

def listsToNumPyMatrixes(samples):
 npArrays=[]
 for (x,y) in samples:
  npArrays.append((listToNumPyMatrix(x),listToNumPyMatrix(y)))
 return npArrays
 
def getRandomWeight(maximum):
 return r.random()*2*maximum-maximum

def getRandomMatrix(size):
 s=''
 (nRows,nColumns)=size
 for row in range(nRows):
  for column in range(nColumns):
   s+=str(getRandomWeight(1))+' '
  if row<nRows-1: s+=';'
 return np.mat(s)
 
def main():
 samplesXOR=[([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]
 nn=NeuralNetwork([2],samplesXOR)
 nn.iterateUntilConvergence()
 for sample in nn.samples:
  (input,output)=sample
  print('input',input,'output',output,'approximate output',nn.feedForward(input))
 
if __name__ == "__main__":
 main()