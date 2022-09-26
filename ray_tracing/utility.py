"""
Created on Fri Feb 22 18:10:02 2020

@author: Sarunas Jurgilas (CCM, 2020)
"""

import numpy as np

def randomPositionFromGaussianCloud(numOfParticles, sigmaR, sigmaA, theta, phi):
    randrVec= []
    for i in range(numOfParticles):
        x = np.random.normal(0, sigmaA)
        y = np.random.normal(0, sigmaR)
        z = np.random.normal(0, sigmaR)
        
        xx = x * np.cos(theta) * np.cos(phi) - y * np.sin(theta) + z * np.cos(theta) * np.sin(phi)
        yy = x * np.sin(theta) * np.cos(phi) + y * np.cos(theta) + z * np.sin(theta) * np.sin(phi)
        zz = - x * np.sin(phi) + z * np.cos(phi) 

        rVec = np.array([xx,yy,zz])
        randrVec.append(rVec)

    return np.array(randrVec)

def uniforms3DSphereSampleGaussian(numberOfSamples):
    randUVec = []
    for i in np.arange(numberOfSamples):
        x = np.random.normal(0,1)
        y = np.random.normal(0,1)
        z = np.random.normal(0,1)

        ex = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
        ey = y / np.sqrt(x ** 2 + y ** 2 + z ** 2)
        ez = z / np.sqrt(x ** 2 + y ** 2 + z ** 2)

        uVec = np.array([ex,ey,ez])
        randUVec.append(uVec)

    return np.array(randUVec)
    