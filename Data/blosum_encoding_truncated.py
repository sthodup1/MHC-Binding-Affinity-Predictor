from __future__ import division
import scipy.io
import numpy as np
import random
import sys, os

hla_type = sys.argv[1]

# Number of segments to break Y into
number_segments = int(sys.argv[2])

mat = scipy.io.loadmat('blosum50')
blosum50 = mat['blosum50']
blosum50 = np.asarray(blosum50)

# Create Blosum encoding dict
keySet = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','*']
valSet = []

for i in range(len(blosum50)):
    valSet.append(blosum50[:][i])

blosum_dict = dict(zip(keySet,valSet))

f1 = open('bdata.2009.mhci.public.1.reduced.txt','r')
f2 = open('bdata.2013.mhci.public.blind.1.reduced.txt','r')

f1_data = f1.readlines()[1:]
f2_data = f2.readlines()[1:]

# Total data set
data = f1_data + f2_data
# Truncate data to only contain the hla_type of interest
temp_data  = []
for line in data:
    split_line = line.strip().split('\t')
    if(split_line[0] == hla_type):
        temp_data.append(line)

data = temp_data
f1.close()
f2.close()

# Find longest sequence length per type
min_length = 100000000;
for line in data:
    split_line = line.strip().split('\t')
    if(len(split_line[1]) < min_length):
        min_length = len(split_line[1])

X = np.ones((len(keySet), min_length, len(data)))
Y = np.zeros((1,len(data)))

for i in range(len(data)):
    line = data[i]
    split_line = line.strip().split('\t')
    seq = split_line[1]

    for j in range(min_length):
        letter = seq[j]
        X[:,j,i] = blosum_dict[letter]
    Y[0,i] = float(split_line[2])

perm = np.random.permutation(len(data))


Y = 1 - np.log(Y)/np.log(np.max(Y))

#Perform Separations and Scalings on Y
Yout = np.zeros((np.shape(Y)[1],number_segments))
for i in range(np.shape(Y)[1]):
    for j in range(1,number_segments+1):
        if Y[0][i] <= j/number_segments and max(Yout[i]) == 0:
            out = np.zeros((1,number_segments))
            out[0][j-1] = 1
            Yout[i][:] = out

# Yout2 = np.zeros((np.shape(Y)[1],3))

# for i in range(np.shape(Y)[1]):
#     if Y[0][i] <= 1/3:
#         Yout2[i][:] = [1, 0, 0]
#     elif Y[0][i] <= 2/3:
#         Yout2[i][:] = [0, 1, 0]
#     else:
#         Yout2[i][:] = [0, 0, 1]

# print np.array_equal(Yout,Yout2)

train_length = int(0.8*len(perm))
test_length = int((len(perm) - train_length)/2)

Xtrain = X[:,:,perm[0:train_length]]
Ytrain = Yout[perm[0:train_length],:]

Xtest = X[:,:,perm[train_length+1:train_length+1+test_length]]
Ytest = Yout[perm[train_length+1:train_length+1+test_length],:]

Xval = X[:,:,train_length+1+test_length+1:]
Yval = Yout[train_length+1+test_length+1:len(perm),:]



os.system('mkdir '+hla_type+'_truncated')

np.save(hla_type+'_truncated/'+'Xtrain',Xtrain)
np.save(hla_type+'_truncated/'+'Ytrain',Ytrain)
np.save(hla_type+'_truncated/'+'Xtest',Xtest)
np.save(hla_type+'_truncated/'+'Ytest',Ytest)
np.save(hla_type+'_truncated/'+'Xval',Xval)
np.save(hla_type+'_truncated/'+'Yval',Yval)

