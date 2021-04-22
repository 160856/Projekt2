import numpy as np
import matplotlib.pyplot as plt
from zadanie15 import zadanie15
from scipy import integrate

"""Czesc 1"""
# -*- coding: utf-8 -*-
# tekst =  "Lorem ipsum dolor sit amet, ..,"
# print (tekst)


# paliwo = 8.5 * (67*0.1 + 124*0.15 + 11.7*0.2)
# autostrada = 124*0.17
# prom = 50
# koszt = paliwo + autostrada + prom
# print (koszt)
# print (koszt/2)
# print (koszt/3)
# print (koszt/4)
# print (koszt/5)

# jakas_zmienna = "To jest jakis napis"
# print (jakas_zmienna)
# jakas_zmienna = 8.5
# print (jakas_zmienna)

# lubie_placki = True
# lista = [1,2,3,4,"napis", True, False, 8.5, 11]
# slownik = {"jablecznik" : "ciasto z jablek", "czy lubisz jablecznik" : True}

# d=[10, 8, 10, 12, 6, 8, 7, 12, 10, 16, 16, 9, 14, 9, 11, 17, 18, 9, 5, 17, 11, 17, 7, 7, 12, 9, 5, 18, 6, 7, 9, 9, 6, 8, 8, 11, 13, 16, 8, 8, 12, 5, 18, 15, 17, 18, 7, 8, 13, 5, 12, 11, 11, 12, 5, 17, 7, 15, 10, 14, 18, 5, 8, 9, 10, 14, 15, 13, 16, 14, 17, 16, 10, 7, 14, 15, 17, 11, 10, 18, 18, 9, 12, 18, 12, 13, 7, 10, 16, 12, 16, 8, 11, 15, 8, 7, 7, 10, 13, 13]
# print (d[16])

# print (d[9])
# print (d[10])
# print (d[11])
# print (d[12])
# print (d[13])
# print (d[14])

# print (d[9:15])
# print (d[:5])
# print (d[-1])
# d[11] = d[11] - 0.1

# d.append(14)
# print (d)

# lista = range(15)
# print (lista)

# for i in range(len(d)):
#     d[i] = d[i]*0.1
    
# print (d)

# for element in [1,2,3,4,5]:
#   print (element)
# print ("zrobimy sobie odstep")
# print (element)

# zmienna = 8
# if zmienna > 5 :
#     print("Zmienna jest wieksza niz 5")

# zmienna = 8
# if zmienna == 15:
#     print ("Zmienna jest rowna 15")
# else:
#     print ("Zmienna nie jest rowna 15")

# zmienna = 8
# if zmienna == 1:
#     print ("Wartosc zmiennej to jeden")
# else:
#     if zmienna == 2:
#         print ("Wartosc zmiennej to dwa")
#     else:
#         print ("Wartosc zmiennej nie jest ani jeden ani dwa")

# x=0
# while x<5:
#     print ("zmienna x wynosi ", x)
#     x+=1
    
# x=0
# while True:
#     print ("To wyswietli sie zawsze gdy warunek jest spelniony")
#     if x==5:
#         print ("zmienna x wynosi ", x)
#         break
#     else:
#         print (x)
#     x+=1 

# def f(x):
#     return (2* (x**3))/8.51

# print (f(5))


# def f(x=0):
#     return (2 * (x**3))/8.51
 
# print (f())
# print (f(5))
"""


""" NUMPY i MATPLOTLIB """
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)

# A = np.array([[1, 2, 3], [7, 8, 9]])
# print(A)

# A = np.array([[1, 2, \
#                 3],
#               [7, 8, 9]])
# print(A)

# v = np.arange(1,7)
# print(v,"\n")
# v = np.arange(-2,7)
# print(v,"\n")
# v = np.arange(1,10,3)
# print(v,"\n")
# v = np.arange(1,10.1,3)
# print(v,"\n")
# v = np.arange(1,11,3)
# print(v,"\n")
# v = np.arange(1,2,0.1)
# print(v,"\n")

# v = np.linspace(1,3,4)
# print(v)
# v = np.linspace(1,10,4)
# print(v)
# v = np.linspace(0,10,3)
# print(v)

# X = np.ones((2,3))
# Y = np.zeros((2,3,4))
# Z = np.eye(2) # np.eye(2,2) np.eye(2,3)
# Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))

#print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
# U = np.block([[A], [X]])
# print(U)

# V = np.block([[
#     np.block([
#         np.block([[np.linspace(1,3,3)],[
#             np.zeros((2,3))]]) ,
#         np.ones((3,1))])
#     ],
#     [np.array([100, 3, 1/2, 0.333])]] )
# print(V)

# print( V[0,2] )
# print( V[3,0] )
# print( V[3,3] )
# print( V[-1,-1] )
# print( V[-4,-3] )

# print( V[3,:] )
# print( V[:,2] )
# print( V[3,0:3] )
# print( V[np.ix_([0,2,3],[0,-1])] )
# print( V[3] )

# Q = np.delete(V, 3, 0)
# print(Q)
# Q = np.delete(V, 2, 1)
# print(Q)
# v = np.arange(1,7)
# print(v)
# print( np.delete(v, 3, 0) )

# print(np.size(v))
# print(np.shape(v))
# print(np.size(V))
# print(np.shape(V))

# A = np.array([[1, 0, 0],
#               [2, 3, -1],
#               [0, 7, 2]] )

# B = np.array([[1, 2, 3],
#               [-1, 5, 2],
#               [2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )


# MM1 = A@B
# print(MM1)
# MM2 = B@A
# print(MM2)

# MT1 = A*B
# print(MT1)
# MT2 = B*A
# print(MT2)

# C = np.linalg.solve(A,MM1)
# print(C) 
# x = np.ones((3,1))
# b =  A@x
# y = np.linalg.solve(A,b)
# print(y)

# PM = np.linalg.matrix_power(A,2) 
# print(PM)
# PT = A**2  
# print(PT)

# # transpozycja
# print(A.T)
# print(A.transpose())
# # hermitowskie sprzezenie macierzy (dla m. zespolonych)
# print(A.conj().T)
# print(A.conj().transpose())



# x=[1,2,3]
# y=[4,6,5]
# plt.plot(x,y)
# plt.show()

# x=np.arange(0.0, 2.0, 0.01)
# y1=np.sin(2.0*np.pi*x)
# y2=np.cos(2.0*np.pi*x)
# y=y1*y2
# l1, = plt.plot(x,y,'b:', linewidth = 3)
# l2,l3 = plt.plot(x,y1,'r*',x,y2,'g--',linewidth=3)
# plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()


""" ZAD 6.3 """

C1 = np.block([[np.linspace(1,5,5)],[np.linspace(5,1,5)]])
C2 = np.block([[2 * np.ones((2, 3))],[np.linspace(-90, -70, 3)]])
C3 = np.block([np.zeros((3, 2)), C2])
C4 = np.block([[C1],[C3]])
A = np.block([C4, 10 * np.ones((5,1))])

print("6.3")
print("A:",A,"\n")



"""ZAD 6.4 """
B = A[1,:] + A[3,:]
print("6.4")
print("B:",B,"\n")



""" ZAD 6.5 """
C = np.array([])
C = np.append(C, max(A[:,0]))
C = np.append(C, max(A[:,1]))
C = np.append(C, max(A[:,2]))
C = np.append(C, max(A[:,3]))
C = np.append(C, max(A[:,4]))
C = np.append(C, max(A[:,5]))

print("6.5")
print("C:",C,"\n")


""" ZAD 6.6 """
D = np.array([])
D = np.delete(B, [0,5])

print("6.6")
print("D:",D,"\n")


""" ZAD 6.7 """
D[D == 4] = 0

print("6.7")
print("D:",D,"\n")


""" ZAD 6.8 """
E = np.array([])
minimum = np.min(C)
maximum = np.max(C)
E = np.delete(C, np.where(C == minimum))
E = np.delete(E, np.where(E == maximum))

print("6.8")
print("E:",E,"\n")


""" ZAD 6.9 """
minimum = np.min(C)
maximum = np.max(C)
print("6.9")
for i in range(np.shape(A)[0]):
    if minimum in A[i,:]:
        if maximum in A[i,:]:
            print(A[i,:])

print("\n")

""" ZAD 6.10 """
print("6.10")
print("Tablicowe: ")
print(D*E,"\n")
print("Wektorowe: ")
print(D@E,"\n")

""" ZAD 6.11 """

def zadanie11(x):
    tab = np.random.randint(0,11,[x, x])
    return tab, np.trace(tab)

print("6.11")
a = int(input('Podaj rozmiar macierzy: '))
print(zadanie11(a),"\n")

""" ZAD 6.12 """

def zadanie12(x):
    np.fill_diagonal(x, 0)
    np.fill_diagonal(np.fliplr(x),0)
    return x

tab2 = np.array([[1,2,3,13],[4,5,6,14],[7,8,9,15],[10,11,12,16]])

print("6.12")
print(zadanie12(tab2),"\n")

""" ZAD 6.13 """

def zadanie13(x):
    suma = 0
    size = np.shape(x)
    for i in range(size[0]):
        if i % 2 == 0:
            suma = suma + np.sum(tab[i, :])
    return suma

print("6.13")
print(zadanie13(tab2),"\n")



""" ZAD 6.14 """

x = np.linspace(-10, 10, 201)
y = lambda x: np.cos(2 * x)
plt.plot(x, y(x), 'g--')
plt.show()

print("6.14")



""" ZAD 6.15 """

x = np.linspace(-10, 10, 201)
tab5 = np.zeros([len(x)])

for i in range(len(x)):
    tab5[i] = zadanie15(x[i])

print("6.15")
plt.plot(x,tab5,'g+',x,y(x),'r--')



""" ZAD 6.17 """

y3=3*(y(x)) + tab5
print("6.17")
plt.plot(x,y3,'b*',x,tab5,'g+',x,y(x),'r--')

""" ZAD 6.18 """

tab6 = np.array([[10,5,1,7],[10,9,5,5],[1,6,7,3],[10,0,1,5]])
tab7 = np.array([[34],[44],[25],[27]])

x= np.linalg.inv(tab6) @ tab7
print("6.18")
print(x,"\n")
print("BABA","\n")


""" ZAD 6.19 """
x = np.linspace(0,2*np.pi,1000000)
y = lambda x:np.sin(x)

calka = integrate.quad(y, 0, 2*np.pi)[0]
print("6.19")
print(calka,"\n")