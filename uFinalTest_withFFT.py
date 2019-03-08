# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:13:55 2019

@author: spenc
"""

import numpy as np
import matplotlib.pyplot as plt
import math
"""

"""

## We set these variables
pi = math.pi
dt = 0.001*pi            # time resolution 
samples = 1000      # number of points in the domain, xsamples
fourierSamples = 1000 # of fourier coefficients we prescribe(-1). For convenience, only use even integers 
p = 2                 # corresponds to u|u|^p

## Some useful inits
domain = np.linspace(0, 2*pi, samples) # x axis, unchanging
hf = int(fourierSamples/2)
#dx = domain[1]
j = 1j



### the dispersion relation w(n), for the schrodinger equation it is = -n^2. Changable for varying equations.
def w(z): 
    return -(z**2) # Note: python evaluates - signs before powers, 

### 0 < x < 2pi, for now u(0,x) = f(x) = 1, -1 for < pi and > pi respectively
def fNot(x):                                    
    if (x >= 0 and x <= pi): 
        return -1
    if (x > pi and x < 2*pi): 
        return 1
    else: return 0
    
### turn our initial function into a list (dtype = np.array)
def fInit(arr):
    workList = np.zeros((samples,), dtype=complex)
    for i in range(0, samples):
        workList[i] = fNot(arr[i])
    return workList

fZero = fInit(domain) #turns our initial function into a list.(funclist)

## OG
# def fInit(arr):
#     workList = [0]*samples
#     for k in range (0, samples):
#         workList[k]=fNot(arr[k])
#     return workList

### Experimenting ifft
def fCoef(funcList):
    outList = np.fft.ifft(funcList)
    return outList

##OG 
## This is inefficient, runs in n^2 time. This almost certainly can be improved, or replaced with FFt.
# def fCoef(funcList):
#     outList = np.zeros((fourierSamples + 1,), dtype=complex) #I think this whole thing can be brought down to single sum, but with 2 arrays(lists)?
#     for n in range(-hf, hf+1): #Hopefully can use fft eventually though        
#         #c_n=(1/2*math.pi)*(funcArray*exp(-j*n*domain)).sum()  #Can see if this works to replace the following 4 lines, might require pandas(?) library...
#         c_n = 0
#         for y in range(0, samples):
#            c_n += funcList[y] * np.exp(-j*n*domain[y]) * dx/(2*math.pi)
#         outList[n] = c_n
#     return outList #approximates fourier coeffecients by approximate sum of the integral. Note that as we are actually
# #                    working with discrete points, the accuracy shouldn't be affected much anyway.
        
### for d/dt u = i*(d/dx)^n u, solution is u
def LinearU(t, fHatArr): 
    currentU = 0
    for n in range(-hf, hf): #evaluates first finite number of terms to approximate the solution.
        currentU += fHatArr[n] * (np.exp(w(n)*t*j, dtype=complex)) * (np.exp(n*domain*j, dtype=complex)) 
    return currentU #an array

def NonlinearU(t, funcList):
    funcArray = np.asarray(funcList)
    arg = t * j * (abs(funcArray)**p)
    return funcArray*np.exp(arg, dtype=complex) 

def totalU(tFinal,fHatInitList):
    currentTime = 0 #Note that eval time is always dt, NOT currentTime. Only used as 'true time' tracker, see below
    fHatList = fHatInitList
    delt = dt
    u2 = LinearU(0, fHatList)
    while (currentTime < tFinal):
        if (currentTime + delt > tFinal): 
            delt = tFinal - currentTime
        u1 = LinearU(delt, fHatList)
        u2 = NonlinearU(delt,u1)
        currentTime += delt
        fHatList = fCoef(u2)
    return u2
    
"""
The split step method allows us to say that u(x,t)=~ U_nonlin*U_lin(x,t) for t<epsilon (t small, * represents function
composition). So we can find u(x,t) for large t by always pretending t=0, and instead updating the 'initial data' with 
the previously calculated values for u(x,dt) over and over. Also with the current fCoef, I have recreated previous graphs
for the seperate solutions accurately, and have gotten a perfect square wave. The only problem is runtime.
"""

##################################TestBits###############################################
time = 0.3
fHatlist = fCoef(fZero)
#Array=NonlinearU(time, fZero)
#Array=LinearU(time,fHatList)
Array = totalU(time,fHatlist)
        
fig = plt.figure()

ax = fig.gca()
fig.set_size_inches(22,16)
ax.set_xlabel("X")
ax.set_ylabel("U")

plt.plot(domain, Array.real)
plt.plot(domain, Array.imag)



#As it it sits, we will have n^4 runtime to create graphs and animations, and something large for box dim.
#can prolly decrease these to n^3 even before FFT, by combining coef and nonlin/lin bits. IDK with fft, but we 
#probably can't do better than n^2log(n)
#Whoever wants to can work on that though
#Also we may be able to imporve our graphs by something similar to PID.
#Nevermind, that removes the behaviour we are looking for in nonlinear times.

    
    