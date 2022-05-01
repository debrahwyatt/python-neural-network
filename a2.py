#!/usr/bin/env python3
import numpy as np
import sys


#######################
#### LORIN SOURA  #####
####      &       #####
#### DEVLIN WYATT #####
#######################


####################
#### Functions #####
####################

#Method used to Parse the file into a dictionary and extract D & N
def parseFile(filename):
   print("Parsing {}...".format(filename))

   f = open(filename)

   N = int(f.readline().rstrip())
   D = int(f.readline().rstrip())

   #skip the header row
   f.readline()

   dic1 = {}

   for i in range(N):
      x = f.readline().rstrip().split('\t')
      dic1[i] = {0: float(x[0]), 1: {}}

      for j in range(D):
         dic1[i][1].setdefault(j,[1,float(x[j+1])])

   print("...{} parsed.\n".format(filename))
   return D, N, dic1


#Used to write the output, still requires coefficient data.
def writeOutput(D, W, filename):
   print("Writing results to {}.csv...".format(filename))

   header = []
   coefficients = []

   for i in range(1, D):
      header.append("w" + str(i))
      coefficients.append(str(W[i-1])) #data goes here
   header.append("w0")
   

   h = "\t".join(header) + "\n"
   c = "\t".join(coefficients)

   #Has to be in tsv format
   f = open("{}.tsv".format(filename), "w")
   d = h+c
   f.write(d)

   print("...{}.tsv saved.\n".format(filename))

#used to calculate W
def weights(N, D, data):

   W = []
   #Need to create lists of data by collumns
   #this should count as transpose (turning a collumn into a row)
   #y wont change so we'll compute it first.
   y = []
   for i in range(N):
      y.append(data[i][0])

   #this is where the main algorithm will start for Question 1
   for j in range(D):
      x = [] 

      for i in range(N):
         x.append(data[i][1][j]) 

      #Need to calculate X^T * X, should give us one value.
      #should just be x_1^2 + x_2^2 + ... + x_n^2
      p=0
      for i in x:
         p = p + i[1]*i[1]
      p = 1/p

      #need to calculate X^T * Y which is X_1*Y_1 + X_2*Y_2 + ... + X_n*Y_n
      l=0
      for i in range(N):
         l = l + x[i][1]*y[i] 

      w = p*l

      #write all values of w to a list
      W.append(w)
   return W


'''
Question 1 is an algorithm that uses the normal equation to learn linear regression models.
The normal Equation is W = (X^T * X)^-1 * X^T * Y
And the linear regression hypothesis is h(x) = w^T * X
'''
def question1(N, D, data):
   print("Starting Question 1...")
   
   W = weights(N, D, data)

   print("...Finished Question 1.\n")
   writeOutput(D+1, W, "Q1")
   return W


'''
Question 2 is on batch gradient descent.
initialize w_0 randomly
Equation is:
  w[j] = w[j] + (nu/N)* (from i=1 to N)(y_i - ^y_i) * x_i[j]
   where ^y_i = w^T * x_i
'''
def question2(N, D, W, data):
   print("Starting Question 2...")

   T = 200 #Epochs
   nu = 0.000001 #learning rate

   W = weights(N, D, data)

   nu/N
   for i in range(N):

      continue

   return None


def cost_function(feature_count, x, y):
   return 0


def question3_a(dictionaries,filename):
   # epoch_limit
   T = 20 
   learning_rate = 0.000001
   batch_size = 1
   bias = 1
   f = open(filename)
   # Get number of data points
   N = int(f.readline().rstrip())
   # Get number of features
   D = int(f.readline().rstrip())
   f.close() #-------------------------------------
   for counter, item in enumerate(dictionaries):
      # Generate a matrix where each data point is a 2D feature vector
      Y = dictionaries[counter][0]

      print(dictionaries[counter][1]) # feature_matrix with ones in first column 
      # The feature set is a dictionary of int labels 0 to 99
      # feature set dictionary contains a list [1, x-value]
      # Access using dictionaries[counter][1]
        
  
      # Get random weight for calculation  
      # loss_function = 1/(2*N)*(y-wt*x) where N is the number of features.
 
      # initialize weight matrix to have the same number of columns as 
      # Loop until we reach T epochs OR loss is sufficently low OR
         # Wgradient = evaluate_gradient(loss, data, W) using:
            # loss function
            # data
            # weight matrix
      # W += -1 * learning_rate * Wgradient 


###############
#### MAIN #####
###############
#############
###########
#########
#######
#####
####
###
##
#

data = {}

filename1 = "data_10k_100.tsv"
filename2 = "data_100k_300.tsv"

D1, N1, data1 = parseFile(filename1)
D2, N2, data2 = parseFile(filename2)

W2 = question1(N2, D2, data2)
W1 = weights(N1, D1, data1)

question2(N1, D1, W1, data1)

#question3(N1, D1, data1)
#question3(N2, D2, data2)


#question3_a(data1,filename1)
#print(dic1[1])
exit(0)
# git clone https://github.com/DevlinWyatt/Seng-474-P2
# git init
# git config user.name "someone"
# git config user.email "someone@someplace.com"
# git add *
# git commit -m "some init msg"
# git push