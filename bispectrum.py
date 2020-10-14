## encoding: utf-8

##############################################################################
##     Algorithm based on:                                                                                     #####
## Komatsu Thesis --> https://wwwmpa.mpa-garching.mpg.de/~komatsu/phdthesis.html                               #####
## Bucher et al  -->  https://arxiv.org/pdf/1509.08107.pdf
## Authors: Jordany Vieira de Melo, Karin Fornazier and Filipe Abdalla
## Email: jordanyv@gmail.com
## Supervisor: F.B. Abdalla
## Date: January 2019
###########################################################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from astropy.io import fits
from math import pi, sin, cos, sqrt, log, floor
from sympy.physics.wigner import gaunt
import os
import sys

###################################################################################
                                                                           #
def gauntB(l1, l2, l3, m1, m2, m3, alm1, alm2, alm3):                            #This function is to calculate the total term on equation (2.3) of the paper https://arxiv.org/pdf/1509.08107.pdf
    b_ell = gaunt(l1,l2,l3,m1,m2,m3)*np.real(alm1*alm2*alm3)   #The l1,l2,l3 and m1,m2,m3 refers to a_lm of the spherical harmonics.
    return b_ell                                                           #The p1,p2,p3 refers to the position in alm vector, which carries the map.
                                                                           #
def alm_position(lmax, la, ma):                                            #This function use the hp.Alm.getidx() to get the position for the variables of the previous function(p1,p2,p3).
    alm_position = hp.Alm.getidx(lmax, la, ma)                             #The variables la and ma refers to the actual values of l and m in loop.
    return alm_position                                                    #
                                                                           #
####################################################################################
def bispectrum_equisize(l0, lmax, path, nome):
    print('Starting the equisize bispectrum calculus for lmax equals to {} and l0 equals to {}'.format(lmax, l0))
    mapa = fits.getdata(path)# Usar para os mapas do Lucas(Foregrounds)
    B_ell=[]                              #In this block we create the vector B_ell to save the bispectrum values for each l.
    alm1 = hp.map2alm(mapa[nome,:])               #Use the function hp.map2alm() to get the a_lm for each map. In this case we use the same map for the 3 a_lm for a equilateral test with same map.
    alm2 = hp.map2alm(mapa[nome,:])               #Retirar o [nome,:] para o caso de mapas thermal noise
    alm3 = hp.map2alm(mapa[nome,:])               #Deixar o [nome,:] e modificar o número 1(dependendo do mapa que quiser usar) dentro do colchete para o trabalho com mapas de Foregrounds
    l1_ell = []
    l2_ell = []
    l3_ell = []
    sum = 0                                    #The sum variable to compute the sum in bispectrum equation.
    l1 = 0                                                                  #Initial values for l1,l2 and l3.
    l2 = 0                                                                  #
    l3 = 0                                                                  #
                                                                        #
    for l1 in range(0, lmax+1):                                             #First loop in l1, because we need to calculate the bispectrum over each l.
        for l2 in range(0, lmax+1):
            for l3 in range(0, lmax+1):
                    if((l1<=l2<=l3) and (abs(l1-l2)<=l3<=abs(l1+l2)) and (l1+l2+l3==l0)):
                        l1_ell.append(int(l1))
                        l2_ell.append(int(l2))
                        l3_ell.append(int(l3))                                                 #
                        for m1 in range(-l1, l1+1):                                         #Now we make the loops over m's, those loops refers to calculate the bispectrum for each l's. Equation (2.3) of the paper https://arxiv.org/pdf/1509.08107.pdf
                            p1 = alm_position(lmax, l1, abs(m1))                            #
                            for m2 in range(-l2, l2+1):                                     #
                                p2 = alm_position(lmax, l2, abs(m2))                        #
                                for m3 in range(-l3, l3+1):                                 #
                                    p3 = alm_position(lmax, l3, abs(m3))                    #
                                    sum += gauntB(l1, l2, l3, m1, m2, m3, alm1[p1], alm2[p2], alm3[p3])       #
                        B_ell.append(sum)                                                   #Here we use the .append() function to save the value of sum in B_ell vector.
                        sum = 0
    np.savetxt('B_ellEqui'+str(nome)+".txt", B_ell, delimiter=',')
    np.savetxt('B_ellTotal123_Equi'+str(nome)+".txt", np.array([B_ell,l1_ell,l2_ell,l3_ell]).T, delimiter=',')
    Belltxt = "B_ellTotal123_Equi"+str(nome)+'.txt'
    pathbe = os.getcwd() + '/' + Belltxt
    print('Finishing the calculus')                                                       #


    ####################################################################################
    # Next block is only for plots the results.
    ###################################################################################

    # A partir daqui começamos a gerar os plots de contorno
    print('Starting contour plots of Equisize Bispectrum')

    #load file and matrix - usou loadtxt para evitar problemas com int ou qq outra coisa nos numeros
    matrixf = np.loadtxt("B_ell_ell_Cube_256_Prior_Equisize_Redshift_"+str(nome)+".txt", delimiter=',')


    #carregou a matriz com os valores de Bell
    matrix_Bell = matrixf[:,0]
    matrix_Bella = matrixf[:,0]
    matrix_Bellt=np.concatenate((matrix_Bell, matrix_Bella), axis=0)
    #print(matrix_Bellt)
    matrix_Bellt=matrix_Bellt.T
    #print(matrix_Bellt)

    #carregou a matriz com os valores de l1
    matrix_ell1 = matrixf[:,1]
    matrix_ell1a = matrixf[:,1]
    matrix_ell1t=np.concatenate((matrix_ell1, matrix_ell1a), axis=0)


    #carregou a matriz com os valores de l2 -l3 aqui precisa de correcao pois os valores são espelhados
    matrix_ell23 = matrixf[:,3]-matrixf[:,2]
    matrix_ell23a = matrixf[:,2]-matrixf[:,3]
    matrix_ell23t=np.concatenate((matrix_ell23, matrix_ell23a), axis=0)
    #print matrix_Bell

    #matrix l1
    Y =matrix_ell1t#matrix_ell1#set the x dimension form matrix shape
    #print Y
    #print (max(Y), min(Y))
    a = Y.max()
    b = np.min(Y)
    #print a,b
    Yarr = np.arange(int(b),int(a)+1)
    #print Yarr

    #matrix l2 - l3

    X =matrix_ell23t#matrix_ell1#set the x dimension form matrix shape
    #print X
    #print max(X), min(X)
    a = X.max()
    b = np.min(X)
    #print a,b
    Xarr = np.arange(int(b),int(a)+1)
    #print Xarr

    sizex = Xarr.size
    #print sizex
    #print Xarr.size
    Bellmatrix = np.zeros((Xarr.size,Yarr.size))
    #print Bellmatrix

    for i in np.arange(Xarr.size):
        for k in np.arange(Yarr.size):
            for j in np.arange(matrix_Bellt.size):
                if ((Yarr[k]==matrix_ell1t[j]) and (Xarr[i]==matrix_ell23t[j])):
                    Bellmatrix[i,k]=matrix_Bellt[j]
    
    #plt.figure(figsize=(10, 8))
    cp = plt.contourf(Xarr, Yarr, Bellmatrix.T)
    #print Xarr.size, Bellmatrix[0,:].size
    #plt.colormap()
    plt.colorbar(cp)
    #plt.gca().invert_yaxis()
    plt.title("Bispectrum Equisize "+str(nome), fontsize=12)
    plt.xlabel('$\ell_2$-$\ell_3$')
    plt.ylabel('$\ell_1$')
    #plt.set_zlim=(-6e-18,6e-18)
    plt.savefig('Bispectrum_Equisize_'+str(nome)+'.png')
    plt.clf()

    print('Finishing function for equisize bispectrum calculus')

###################################################################################################################################################

def bispectrum_isoceles(lmax, path, nome):
    print('Starting the isoceles bispectrum calculus for lmax equals to {}'.format(lmax))
    mapa = fits.getdata(path)# Usar para os mapas do Lucas(Foregrounds)
    B_ell=[]                              #In this block we create the vector B_ell to save the bispectrum values for each l.
    alm1 = hp.map2alm(mapa[nome,:])               #Use the function hp.map2alm() to get the a_lm for each map. In this case we use the same map for the 3 a_lm for a equilateral test with same map.
    alm2 = hp.map2alm(mapa[nome,:])               #Retirar o [nome,:] para o caso de mapas thermal noise
    alm3 = hp.map2alm(mapa[nome,:])               #Deixar o [nome,:] e modificar o número 1(dependendo do mapa que quiser usar) dentro do colchete para o trabalho com mapas de Foregrounds
    l1_ell = []
    l2_ell = []
    l3_ell = []
    sum = 0                                                                #
    l1 = 0                                                                  #Initial values for l1,l2 and l3.
    l2 = 0                                                                  #
    l3 = 0                                                                  #
                                                                            #
    for l1 in range(0, lmax+1):                                             #First loop in l1, because we need to calculate the bispectrum over each l.
        for l2 in range(0, lmax+1):
            for l3 in range(0, lmax+1):
                    if((l1<=l2<=l3) and (abs(l1-l2)<=l3<=abs(l1+l2)) and (l1==l2)):
                        l1_ell.append(int(l1))
                        l2_ell.append(int(l2))
                        l3_ell.append(int(l3))                                                 #
                        for m1 in range(-l1, l1+1):                                         #Now we make the loops over m's, those loops refers to calculate the bispectrum for each l's. Equation (2.3) of the paper https://arxiv.org/pdf/1509.08107.pdf
                            p1 = alm_position(lmax, l1, abs(m1))                            #
                            for m2 in range(-l2, l2+1):                                     #
                                p2 = alm_position(lmax, l2, abs(m2))                        #
                                for m3 in range(-l3, l3+1):                                 #
                                    p3 = alm_position(lmax, l3, abs(m3))                    #
                                    sum += gauntB(l1, l2, l3, m1, m2, m3, alm1[p1], alm2[p2], alm3[p3])       #
                        B_ell.append(sum)                                                   #Here we use the .append() function to save the value of sum in B_ell vector.
                        sum = 0
    np.savetxt('B_ellIso'+str(nome)+".txt", B_ell, delimiter=',')
    np.savetxt('B_ellTotal123_Iso'+str(nome)+".txt", np.array([B_ell,l1_ell,l2_ell,l3_ell]).T, delimiter=',')
    Belltxt = "B_ellTotal123_Iso"+str(nome)+".txt"
    print('Finishing the calculus')                                                             #
                                                                 #
                                                                            #
    ####################################################################################
    # Next block is only for plots the results.
    ###################################################################################

    print('Starting contour plots of Isoceles Bispectrum')
    matrixfi = np.loadtxt("B_ell_ell_Isosceles_prior_maps_Cubo_Jordany_128_Redshift_"+str(nome)+".txt", delimiter=',')

    #Load Bell values
    matrix_Belli = matrixfi[:,0]
    matrix_Bellia = matrixfi[:,0]
    matrix_Bellit=np.concatenate((matrix_Belli, matrix_Bellia), axis=0)
    matrix_Bellit=matrix_Bellit.T

    #Load values ell3
    matrix_elli3 = matrixfi[:,3]
    matrix_elli3a = matrixfi[:,3]
    matrix_elli3t=np.concatenate((matrix_elli3, matrix_elli3a), axis=0)

    #load values of ell1 and ell2. These values should be mirrored for triangle contour
    matrix_elli12 = matrixfi[:,1]
    matrix_elli12a = matrixfi[:,2]
    matrix_elli12t=np.concatenate((matrix_elli12, matrix_elli12a), axis=0)

    #matrix l3
    #set the x dimension form matrix shape
    Yi =matrix_elli3t
    a = Yi.max()
    b = np.min(Yi)
    Yiarr = np.arange(int(b),int(a)+1)

    #matrix_ell1 = matrixell2
    #set the x dimension form matrix shape
    Xi =matrix_elli12t
    a = Xi.max()
    b = np.min(Xi)
    Xiarr = np.arange(int(b),int(a)+1)
    sizex = Xiarr.size

    Bellmatrixi = np.zeros((Xiarr.size,Yiarr.size))
    for i in np.arange(matrix_Bellit.size):
        Bellmatrixi[int(matrix_elli12t[i]),int(matrix_elli3t[i])]=matrix_Bellit[i]

    cp = plt.contourf(Xiarr, Yiarr, Bellmatrixi.T,cmap=cm.viridis )

    plt.colorbar(cp)
    plt.title("Bispectrum Isosceles"+str(nome))
    plt.xlabel('$\ell_1=\ell_2$')
    plt.ylabel('$\ell_3$')

    plt.savefig('Bispectrum_Isosceles_'+str(nome)+".png")
    plt.clf()

    print('Finishing function for isoceles bispectrum calculus')
