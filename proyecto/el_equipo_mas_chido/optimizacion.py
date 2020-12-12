import numpy as np
import time
import csv
import random
import matplotlib.pyplot as plt

class optimizacion:
    def __init__(self,function,max_iters=-1,tol=1):
        self.f = function
        self.max_iters=300 if max_iters==-1 else max_iters
        self.tol=np.float64(1e-10) * np.ones(np.size(x)) if tol==1 else tol

    def gradiente(self,x):
        x = x.astype(np.float64)
        h = np.float64(1e-4)
        k = 1/(2*h)
        n = x.shape[0]
        grad = np.zeros(n).astype(np.float64)
        for i in range(n):
            aux1 = np.copy(x)
            aux2 = np.copy(x)
            aux1[i] = aux1[i]+h
            aux2[i] = aux2[i]-h
            grad[i] = self.f(aux1) - self.f(aux2)
            grad[i] = grad[i]*k
        return grad
    
    def hessiana(self,x):
        x = x.astype(np.float64)
        h = np.float64(1e-2)
        k = 1/(h**2)
        n = x.shape[0]
        hess = np.zeros((n,n)).astype(np.float64)
        for i in range(n):
            for j in range(i+1):
                if i == j:
                    aux1 = np.copy(x)
                    aux2 = np.copy(x)
                    aux1[i] = aux1[i]+h
                    aux2[i] = aux2[i]-h
                    hess[i,j] = self.f(aux1) + self.f(aux2) - 2*self.f(x)
                    hess[i,j] = hess[i,j]*k
                else:
                    aux1 = np.copy(x)
                    aux2 = np.copy(x)
                    aux3 = np.copy(x)
                    aux1[i] = aux1[i]+h
                    aux1[j] = aux1[j]+h
                    aux2[i] = aux2[i]+h
                    aux3[j] = aux3[j]+h
                    hess[i,j] = self.f(aux1) - self.f(aux2) - self.f(aux3) + self.f(x)
                    hess[i,j] = hess[i,j]*k
                    hess[j,i] = hess[i,j]
        return hess
    
    def es_optimo(self,x):
        grad = abs(self.gradiente(x))
        if all(grad < self.tol):
            hess = self.hessiana(x)
            if all (np.linalg.eigh(hess)[0] >= 0):
                optimo = True
            else:
                optimo = False
        else:
            optimo = False
        return optimo
    
    def wolfe(self,x,p,alpha,c1,c2):
        aux = x + alpha*p
        producto = np.dot(self.gradiente(x),p)
        cond1=False
        cond2=False
        if self.f(aux) <= self.f(x) + c1*alpha*producto:
            cond1=True
        if np.dot(self.gradiente(aux),p) >= c2*producto:
            cond2=True
        return cond1 and cond2
    
    def alpha(self,x,p,c1,c2,rho):
        a = 1
        i = 1
        M = 500
        while i<M and self.wolfe(x,p,a,c1,c2)==False:
            a = rho*a
            i = i+1
        if i==M:
            print("iteraciones maximas alcanzadas en Wolfe")
        return a
    
    def volver_pd(self,x):
        if np.all(np.linalg.eigvals(x) > 0) == True:
            return x
        else:
            e = abs(np.linalg.eigvals(x))
            l = min(e) + 3*np.finfo(float).eps
            E = np.identity(len(x))
            x = x+(l*E)
        return x
    
    def busqueda_lineal_newton(self,x,modificada = False,comparacion=False):
        k=0
        c1 = 0.0001
        c2 = 0.9
        rho = 0.6
        a = 1
        while k<self.max_iters and self.es_optimo(x)==False:
            B = self.hessiana(x)
            g = -self.gradiente(x)
            p = np.linalg.solve(B,g)
            if modificada == True:
                B = self.volver_pd(B)
                a = self.alpha(x,p,c1,c2,rho)
            x = x+a*p
            k=k+1
        if k==self.max_iters:
            if modificada:
                print ("iteraciones maximas alcanzadas en busqueda lineal modificada")
            else:
                print ("iteraciones maximas alcanzadas en busqueda lineal")
        elif comparacion:
            if modificada:
                print("Busqueda Lineal Newton modificada tardó ",k," iteraciones en llegar al óptimo")
            else:
                print("Busqueda Lineal Newton tardó ",k," iteraciones en llegar al óptimo")

        return x
    
    def BFGS (self,xk,comparacion=False):
        I=np.identity(np.size(xk))
        H=I
        c1 = 10**(-4)
        c2 = 0.9
        rho = 0.6
        k=0
        while k<self.max_iters and self.es_optimo(xk)==False:
            #H=self.volver_pd(H)
            pk=np.linalg.lstsq(-H,self.gradiente(xk),rcond=None)
            pk=pk[0]
            if k==0:
                a=1
            else:
                a=self.alpha(xk, pk, c1, c2, rho)
            xk1=xk+a*pk
            sk=xk1-xk
            yk=self.gradiente(xk1)-self.gradiente(xk)
            rho_k=1/(np.dot(yk,sk))
            A=(I-(rho_k*np.dot(sk,yk)))
            B=(I-(rho_k*np.dot(yk,sk)))
            C=rho_k*np.dot(sk,sk)
            #H=(I-(rho_k*np.dot(sk,yk)))*H*(I-(rho_k*np.dot(yk,sk)))+rho_k*np.dot(sk,sk)
            H=np.matmul(A,H)
            H=np.matmul(H,B)
            H=H+C
            xk=xk1
            k=k+1
        if k==self.max_iters:
            print ("iteraciones máximas alcanzadas en BFGS")
        elif comparacion:
            print("BFGS tardó ",k," iteraciones en llegar al óptimo")
        return xk
    
    def algoritmo_newton(self,x,comparacion=False):
        k=0
        c1 = 0.1
        c2 = 0.8
        rho = 0.9
        a = 1
        while k<self.max_iters and self.es_optimo(x)==False:
            B = self.hessiana(x)
            g = -self.gradiente(x)
            p = np.linalg.solve(B,g)
            a = self.alpha(x,p,c1,c2,rho)
            x = x+a*p
            k=k+1
        if k==self.max_iters:
            print ("iteraciones máximas alcanzadas en Newton")
        elif comparacion:
            print("Newton tardó ",k," iteraciones en llegar al óptimo")
        return x
    
    def comparacion(self,x):
        start_time = time.time()
        x_N=self.algoritmo_newton(x,True)
        f_N=self.f(x_N)
        print("Mínimo de Newton: ",x_N," y es: ",f_N)
        t_N=(time.time() - start_time)
        print("Tardó ",t_N)
        start_time = time.time()
        x_bln=self.busqueda_lineal_newton(x,modificada=False,comparacion=True)
        f_bln=self.f(x_bln)
        print("Mínimo de búsqueda lineal de Newton: ",x_bln,", y es: ",f_bln)
        t_BLN=(time.time() - start_time)
        print("Tardó ",t_BLN)
        start_time = time.time()
        x_blnm=self.busqueda_lineal_newton(x,modificada=True,comparacion=True)
        f_blnm=self.f(x_blnm)
        print("Mínimo de búsqueda lineal de Newton modificada: ",x_blnm,", y es: ",f_blnm)
        t_BLNM=(time.time() - start_time)
        print("Tardó ",t_BLNM)
        start_time = time.time()
        x_BFGS=self.BFGS(x,True)
        f_BFGS=self.f(x_BFGS)
        print("Mínimo de BFGS: ",x_BFGS," y es: ",f_BFGS)
        t_BFGS=(time.time() - start_time)
        print("Tardó ",t_BFGS)
        
        y=np.array([t_N,t_BLN,t_BLNM,t_BFGS])
        x=np.array([1,2,3,4])
        #y = np.array([0.650, 0.660, 0.675, 0.685])
        my_xticks = ['Newton', 'BLN', 'BLNM', 'BFGS']
        plt.xticks(x, my_xticks)
        #plt.yticks(np.arange(y.min(), y.max(), 0.005))
        plt.plot(x, y)
        plt.grid(axis='y', linestyle='-')
        plt.title("Tiempos")
        plt.ylabel("tiempo en segundos")
        plt.xlabel("Métodos")
        plt.show()
        
        # plt.xlim(min(x), max(x))
    # plt.ylim(min(y), max(y))
        
# Función de Rosenbrock

# a=1
# b=100    
# fR = lambda x: (a-x[0])**2 + b*(x[1]-x[0]**2)**2
# x = np.array([0,0])
# optR=optimizacion(fR,max_iters=1000)
# optR.comparacion(x)

# Prueba con diversos parámetros

# for i in range(1,11,4):
#     print("para a=",i)
#     a=i
#     b=100    
#     fR = lambda x: (a-x[0])**2 + b*(x[1]-x[0]**2)**2
#     x = np.array([0,0])
#     optR=optimizacion(fR,max_iters=100)
#     optR.comparacion(x)

# Problema cámaras

datos=np.loadtxt(open("crime_data.txt", "rb"), delimiter=",", skiprows=1,usecols=range(3,5))
n=800

x = np.ones(2*n)
for i in range(0,2*n):
    x[i]=x[i]*random.uniform(1,20)
def fCrimen(x):
    x = x.reshape(2,n)
    r=0
    for i in range(0,n):
        for j in range(0,n):
            r=r+(np.linalg.norm(x[:,i]-datos[j]))**2
    for ii in range(0,n):
        for jj in range(0,n):
            if ii != jj:
                r=r+1/((np.linalg.norm(x[:,ii]-x[:,jj]))**2)
    return r







print("Valor inicial: ",fCrimen(x))
optC=optimizacion(fCrimen,max_iters=2)
r=optC.BFGS(x)
print("El valor mínimo obtenido fue de ",fCrimen(r))
