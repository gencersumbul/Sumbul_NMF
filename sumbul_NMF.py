import numpy as np
from operator import add

class NMF:
    def __init__(self,V,num_of_basis, initialize = 'random_acol',p=0):
        self.V = np.array(V)
        self.W=np.zeros((len(V),num_of_basis))
        self.H=np.ones((num_of_basis,len(V[0])))
        self.num_of_basis = num_of_basis
        self.row_dim = len(V)
        self.column_dim = len(V[0])
        self.errors=[]
        if initialize=='random':
            self.random_initiate_factors()
        elif initialize=='random_acol':
            if p==0:
                self.randomAcol_initiate_factors(num_of_basis)
            else:
                self.randomAcol_initiate_factors(p)
        elif initialize=='multiple_random':
            self.random_initiate_factors()

    def random_initiate_factors(self):
        self.W=np.random.random([self.row_dim, self.num_of_basis])
        self.H=np.random.random([self.num_of_basis, self.column_dim])

    def randomAcol_initiate_factors(self,p):
        for i in range(self.num_of_basis):
            columns = np.random.randint(self.column_dim,size=p)
            self.W[:,i] = np.mean(self.V[:,columns],axis=1)
        self.H=np.random.random([self.num_of_basis, self.column_dim])

    def calc_E_distance(self):
    	product = np.dot(self.W,self.H)
    	E = np.sum(np.square(np.absolute(np.subtract(self.V,product))))
        self.errors.append(E)        

    def calc_KL_divergence(self):
    	product = np.dot(self.W,self.H)
    	KL_divergence = 0.0
    	for i in range(self.row_dim):
    		for j in range(self.column_dim):
    			KL_divergence+=	np.multiply(self.V[i,j],np.log(np.true_divide(self.V[i,j],product[i,j])))-self.V[i,j]+product[i,j]
        self.errors.append(KL_divergence)

    def calc_IS_divergence(self):
     	product = np.dot(self.W,self.H)
    	IS_divergence = 0.0
    	for i in range(self.row_dim):
    		for j in range(self.column_dim):
    			IS_divergence+=	np.true_divide(self.V[i,j],product[i,j])-np.log(np.true_divide(self.V[i,j],product[i,j]))-1
    	self.errors.append(IS_divergence)   	

    def E_update_factors(self):
    	H_numerator = np.dot(np.transpose(self.W),self.V)
    	H_denominator = np.dot(np.dot(np.transpose(self.W),self.W),self.H)	

    	for i in range(self.num_of_basis):
    		for j in range(self.column_dim):
    			self.H[i,j]=np.multiply(self.H[i,j],np.true_divide(H_numerator[i,j],H_denominator[i,j]))

    	W_numerator = np.dot(self.V,np.transpose(self.H))
    	W_denominator = np.dot(np.dot(self.W,self.H),np.transpose(self.H))

    	for i in range(self.row_dim):
    		for j in range(self.num_of_basis):
    			self.W[i,j]=np.multiply(self.W[i,j],np.true_divide(W_numerator[i,j],W_denominator[i,j]))

    def KL_update_factors(self):
    	product = np.dot(self.W,self.H)
    	self.H = np.multiply(self.H,np.true_divide(np.dot(np.transpose(self.W),np.true_divide(self.V,product)),np.dot(np.transpose(self.W),np.ones((self.row_dim,self.column_dim)))))
    	product = np.dot(self.W,self.H)
    	self.W = np.multiply(self.W,np.true_divide(np.dot(np.true_divide(self.V,product),np.transpose(self.H)),np.dot(np.ones((self.row_dim,self.column_dim)),np.transpose(self.H)))) 

    def normalize_basis(self):
    	for i in range(self.num_of_basis):
    		self.W[:,i] = np.true_divide(self.W[:,i],np.sum(self.W[:,i]))

    def E_update_factors_with_sparseness(self,Lambda):
        self.normalize_basis()  	
    	product = np.dot(self.W,self.H)
    	for i in range(len(self.H[0])):
            for j in range(len(self.H)):
                self.H[j,i]=np.multiply(self.H[j,i],np.true_divide(np.dot(self.V[:,i],self.W[:,j]),np.dot(product[:,i],self.W[:,j])+Lambda))
    	product = np.dot(self.W,self.H)

    	for j in range(len(self.W[0])):
    		W_numerator=0.0
    		W_denominator=0.0
    		for i in range(len(self.H[0])):
    			W_numerator+=np.multiply(self.H[j,i],map(add,self.V[:,i],np.dot(np.dot(product[:,i],np.transpose(self.W[:,j])),self.W[:,j])))
    			W_denominator+=np.multiply(self.H[j,i],map(add,product[:,i],np.dot(np.dot(self.V[:,i],np.transpose(self.W[:,j])),self.W[:,j])))
    		self.W[:,j]=np.multiply(self.W[:,j],np.true_divide(W_numerator,W_denominator))

    def IS_update_factors(self):
    	product = np.dot(self.W,self.H)
    	self.H = np.multiply(self.H,np.true_divide(np.dot(np.transpose(self.W),np.true_divide(self.V,np.square(product))),np.dot(np.transpose(self.W),np.true_divide(np.ones((self.row_dim,self.column_dim)),product))))
    	product = np.dot(self.W,self.H)
    	self.W = np.multiply(self.W,np.true_divide(np.dot(np.true_divide(self.V,np.square(product)),np.transpose(self.H)),np.dot(np.true_divide(np.ones((self.row_dim,self.column_dim)),product),np.transpose(self.H))))

    def check_change(self,old,new):
    	if old==new:
    		return False
    	else:
    		return True

    def get_basis(self):
    	return self.W

    def get_errors(self):
        return self.errors

    def get_weights(self):
    	return self.H

    def multiple_run_NMF(self,cost='E'):
        e=[]
        result = np.inf
        for i in range(10):
            self.errors = []
            self.run_NMF(cost)
            if self.errors[len(self.errors)-1]<result:
                result = self.errors[len(self.errors)-1]
                e = self.errors
                h = self.H
                w = self.W
            self.random_initiate_factors()
        self.errors = e
        self.H = h
        self.W = w

    def run_NMF(self,cost='E',Lambda=0):
        if cost=='E':
            if Lambda==0:
                for i in range(1000):
                    self.E_update_factors()
                    self.calc_E_distance()
            else:
                for i in range(1000):
                    self.E_update_factors_with_sparseness(Lambda)
        elif cost=='KL':
            for i in range(1000):
                self.KL_update_factors()
                self.calc_KL_divergence()
        elif cost=='IS':
            for i in range(1000):
                self.IS_update_factors()
                self.calc_IS_divergence()
