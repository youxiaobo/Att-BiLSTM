# coding:utf8
import numpy as np
import ipdb

# displacement in x direction
def getDeltaX(x):
    deltaX = np.diff(x)
    return deltaX

# displacement in y direction
def getDeltaY(y): 
    deltaY = np.diff(y)
    return deltaY

# Instantaneous speed
def getVi(x,y):
    deltaX = np.diff(x)    
    deltaY = np.diff(y)
    vi = np.sqrt(np.square(deltaX) + np.square(deltaY))
    return vi

# Instantaneous angle
def getAi(x,y):
    
    deltaX = np.diff(x)    
    deltaY = np.diff(y)
    length = deltaX.shape[0]
    ai = np.zeros([length,1])
    for i in range(0,length):
        if deltaX[i] == 0:
            ai[i] = np.pi / 2
        else:
            ai[i] = np.arctan(deltaY[i]/deltaX[i])
    return ai

# Total distance traveled
def getDtot(x,y,shift):
    
    deltaX = np.diff(x)    
    deltaY = np.diff(y)     
    di = np.sqrt(np.square(deltaX) + np.square(deltaY))
    length = di.shape[0]
    dtot = np.zeros([length,1])
    # calculate the sum of di with nearest frame length equal to 2*shift+1 
    for i in range(0,length):
        startI = i - shift
        endI = i + shift
        if startI < 0:
           startI = 0
        if endI > length-1:
           endI = length-1
        dtot[i] = np.sum(di[startI:endI+1])
    return dtot

# Net distance traveled
def getDnet(x,y,shift):
    length = x.shape[0]-1
    dnet = np.zeros([length,1])
    # calculate the network distance with nearest frame length equal to 2*shift+2(corresponding to dtot) 
    for i in range(0,length):
        startI = i - shift
        endI = i + shift + 1
        if startI < 0:
           startI = 0
        if endI > length:
           endI = length
        dnet[i] = np.sqrt(np.square(x[endI]-x[startI])+np.square(y[endI]-y[startI]))
    return dnet

# Confinement ratio
def getRcon(x,y,shift):
    dtot = getDtot(x,y,shift)
    dnet = getDnet(x,y,shift)
    rcon = dnet / dtot
    return rcon

# Time averaged mean square displacement
def getMSD(x,y,shift,n):
    length = x.shape[0]-1
    msd = np.zeros([length,1])
    nlength = x.shape[0] - n
    dn = np.zeros([nlength,1])
    for i in range(0,nlength):
        dn[i] = np.sqrt(np.square(x[i+n]-x[i])+np.square(y[i+n]-y[i]))
    dnSquare = np.square(dn)

    # pay attention to the startI and endI with different n
    for i in range(0,length):
        if n % 2 == 0:
           startI = i - (shift - ((n - 1) // 2))
           endI = i + (shift - ((n - 1) // 2)) - 1
        else:
           startI = i - (shift - (n // 2))
           endI = i + (shift - (n // 2))
        if startI < 0:
            startI = 0
        if endI > dnSquare.shape[0]-1:
            endI = dnSquare.shape[0]-1
        msd[i] = np.mean(dnSquare[startI:endI+1]) 
    return msd

# MSD ratio
def getMSDRatio(x,y,shift,n1,n2):
    msd1 = getMSD(x,y,shift,n1)
    msd2 = getMSD(x,y,shift,n2)
    # avoid divide zero
    msd2[msd2==0] = 0.0001
    msdRatio = msd1/msd2 - n1/n2
    return msdRatio


# gyration tensor,used for asymmetry and kurtosis
def getGyrationTensor(x,y,shift):
    gyrationT = []
    length = x.shape[0]-1
    for i in range(0,length):
        startI = i -shift
        endI = i + shift + 1
        if startI < 0:
            startI = 0
        if endI > length:
            endI = length
        trackX = x[startI:endI+1]
        trackY = y[startI:endI+1]
        meanX = np.mean(trackX)
        meanY = np.mean(trackY)
        t11 = np.mean(np.square(trackX - meanX))
        t12 = np.mean(np.multiply((trackX-meanX),(trackY-meanY)))
        t21 = t12
        t22 = np.mean(np.square(trackY - meanY))
        T = np.array([[t11,t12],[t21,t22]])
        gyrationT.append(T)
    return gyrationT

# asymmetry
def getA(x,y,shift):
    gyrationT = getGyrationTensor(x,y,shift)
    length = len(gyrationT)
    A = np.zeros([length,1])
    for i in range(length):
        T = gyrationT[i]
        eigenValues,eigenVectors = np.linalg.eig(T)
        lambda1 = eigenValues[0]
        lambda2 = eigenValues[1]
        tmp = 1-np.square(lambda1-lambda2)/(2*(lambda1+lambda2))
        #if tmp <= 0:
        #    print('i={},tmp={}'.format(i,tmp))
            

        #A[i] = -np.log10(tmp)
        A[i] = tmp
    return A

# Kurtosis
def getK(x,y,shift):
    length = x.shape[0]-1
    k = np.zeros([length,1])
    for i in range(0,length):
        startI = i -shift
        endI = i + shift + 1
        if startI < 0:
            startI = 0
        if endI > length:
            endI = length
        trackX = x[startI:endI+1]
        trackY = y[startI:endI+1]
        meanX = np.mean(trackX)
        meanY = np.mean(trackY)
        t11 = np.mean(np.square(trackX - meanX))
        t12 = np.mean(np.multiply((trackX-meanX),(trackY-meanY)))
        t21 = t12
        t22 = np.mean(np.square(trackY - meanY))
        T = np.array([[t11,t12],[t21,t22]])
        eigenValues,eigenVectors = np.linalg.eig(T)
        
        # sort eigenValues
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        dominantV = eigenVectors[:,0]
        N = 2*shift+2
        xp = np.full((N,1),np.nan)
        jj = 0
        for j in range(startI,endI+1):
            xp[jj] = np.dot(np.array([x[j],y[j]]),dominantV.reshape(2,1))
            jj = jj+1
        # expect nan
        xp = xp[np.isnan(xp)==False].reshape(-1,1)
        xpMean = np.mean(xp)
        xpStd = np.std(xp)
        k[i] = np.mean(((xp-xpMean)**4)/(xpStd**4))
    return k
    
# Efficiency
def getE(x,y,shift):
    deltaX = np.diff(x)    
    deltaY = np.diff(y)     
    di = np.sqrt(np.square(deltaX) + np.square(deltaY))
    length = x.shape[0]-1
    E = np.zeros([length,1])
    dtotSquare = np.zeros([length,1])
    dnetSquare = np.zeros([length,1])
    for i in range(0,length):
        startI = i - shift
        endI = i + shift
        if startI < 0:
           startI = 0
        if endI > length-1:
           endI = length-1
            
        dtotSquare[i] = np.sum(np.square(di[startI:endI+1]))
        dnet = np.sqrt(np.square(x[endI+1]-x[startI])+np.square(y[endI+1]-y[startI]))
        dnetSquare[i] = np.square(dnet)
        E[i] = dnetSquare[i]/((2*shift+1)*dtotSquare[i])
    return E

# Fractal dimension
def getDf(x,y,shift):
    
    deltaX = np.diff(x)    
    deltaY = np.diff(y)     
    di = np.sqrt(np.square(deltaX) + np.square(deltaY))
    length = di.shape[0]
    Df = np.zeros([length,1])
    for i in range(0,length):
        
        startI = i - shift
        endI = i + shift
        if startI < 0:
           startI = 0
        if endI > length-1:
           endI = length-1
        L = np.sum(di[startI:endI+1])
        d = np.max(di[startI:endI+1])
        N = 2*shift+1
        Df[i] = np.log10(N)/np.log10(N*d/L)
    return Df











