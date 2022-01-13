#!/usr/bin/env python
# coding: utf-8

# In[17]:


# library for trigonometry and square-root only
import math
import matplotlib.pyplot as plt


# In[18]:


class QuantumProgram:
    
    # Identity matrix
    I=[[1,0],[0,1]]
    
    # Zero  vector
    zero_vector=[[1],[0]]
    
    # Gates applied
    job=[]
    
    # defining no of qubits, unitary matrix, initial tensored product 
    def __init__(self,no_of_qubits):
        
        # defining no of qubits
        self.no_of_qubits=no_of_qubits
        
        # list containing the basis states according to no_of_qubits
        # for 2 qubits basis = ['00','01','10','11']
        basis=[]
        
        # Identity tensor for initial unitary matrix
        mat= [[1]]
        
        # Identity tensor for initial state_vector
        state_vector=[[1]]
        
        for i in range(self.no_of_qubits):
            mat=self.Tensor_matrix(mat,self.I)
            state_vector=self.Tensor_matrix(state_vector,self.zero_vector)
            
        self.unitary=mat
        self.state_vector=state_vector
        
        for i in range(2**self.no_of_qubits):
            basis.append(self.basis_state(i))
        self.basis=basis
        
        
        
    # method which converts decimal to binary and length of bit string is no_of_qubits    
    def basis_state(self,n):
        a=[]
        if n>0:
            while(n>0):
                dig=n%2
                a.append(dig)
                n=n//2
        else:
            a=[0]
        a.reverse()
        while(len(a)!=self.no_of_qubits):
            a=[0]+a
        str_=''
        for i in a:
            str_+=str(i)
        return str_
    
    def createMatrix(self,n):
        mat = []
        for i in range(n):
            rowList = []
            for j in range(n):
                rowList.append(0)
            mat.append(rowList)

        return mat
    
    
    def binaryToDecimal(self,binary):
     
        binary1 = binary
        decimal, i, n = 0, 0, 0
        while(binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary//10
            i += 1
        return decimal
    
    # Method to multiply vectors with matrix
    def vector_mat_mul(self,v,G):
        result=[]
        for i in range(len(G[0])):
            total=0
            for j in range(len(v)):
                total+= v[j][0]*G[i][j]
            result.append([total])
        return result
    
    # Method to multiply matrix with matrix
    def mat_mat_mul(self,mat1,mat2):
        res = [[0 for x in range(len(mat2[0]))] for y in range(len(mat1))]  
  
     
        for i in range(len(mat1)): 
            for j in range(len(mat2[0])): 
                for k in range(len(mat2)): 

                    # resulted matrix 
                    res[i][j] += mat1[i][k] * mat2[k][j] 
        return res
    
    # method to calculate the tensor product of 2 vectors
    
    def Tensor_vector(self, A, B):
        rowa=len(A)
        rowb=len(B)
        C = [[0] for i in range(rowa * rowb)]

        # i loops till rowa
        for i in range(0, rowa):

            # k loops till rowb
            for j in range(0, rowb):
                
                C[2*i + j][0] = A[i][0]* B[j][0]
                    
        return C 
    
    
    # method to calculate the tensor product of 2 matrices
    
    def Tensor_matrix(self, A , B ):
        if type(A[0])==int:
            cola=0
        else:
            cola = len(A[0])
        rowa = len(A)
        if type(B[0])==int:
            colb=0
        else:
            colb = len(B[0])
        rowb = len(B)

        C = [[0 for j in range(cola * colb)] for i in range(rowa * rowb)]
        
        # i loops till rowa
        for i in range(0, rowa):

            # j loops till cola
            for j in range(0, cola):

                # k loops till rowb
                for k in range(0, rowb):

                    # l loops till colb
                    for l in range(0, colb):

                        # Each element of matrix A is
                        # multiplied by whole Matrix B
                        # respectively and stored in Matrix C
                        C[2*i + k][2*j + l] = A[i][j] * B[k][l]
                       
        return C 
    
    # Single-qubit H-gate
    
    def h(self,qubit_index):
        H_operator=[[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]]
        pdt=[[1]]
        for i in range(self.no_of_qubits-1,-1,-1):
            
            if i!=qubit_index:
                pdt= self.Tensor_matrix(pdt,self.I)
            else:
                pdt=self.Tensor_matrix(pdt,H_operator)
        self.unitary = self.mat_mat_mul(pdt,self.unitary)
        
        return 0
    
        
    # Single-qubit X-gate
    
    def x(self,qubit_index):
        x_operator=[[0,1],[1,0]]
        pdt=[[1]]
        for i in range(self.no_of_qubits-1,-1,-1):
            
            if i!=qubit_index:
                pdt= self.Tensor_matrix(pdt,self.I)
            else:
                pdt=self.Tensor_matrix(pdt,x_operator)
        self.unitary = self.mat_mat_mul(pdt,self.unitary)
        
        return 0
        
    # Single-qubit Z-gate
    def z(self,qubit_index):
        z_operator=[[1,0],[0,-1]]
        pdt=[[1]]
        for i in range(self.no_of_qubits-1,-1,-1):
            
            if i!=qubit_index:
                pdt= self.Tensor_matrix(pdt,self.I)
            else:
                pdt=self.Tensor_matrix(pdt,z_operator)
        self.unitary = self.mat_mat_mul(pdt,self.unitary)
        
        return 0
        
    # Single-qubit Rotation gate, the angle should be in radians
    
    def rotation(self,angle,qubit_index):
        operator=[[math.cos(angle), -math.sin(angle)],[math.sin(angle),math.cos(angle)]]
        pdt=[[1]]
        for i in range(self.no_of_qubits-1,-1,-1):
            
            if i!=qubit_index:
                pdt= self.Tensor_matrix(pdt,self.I)
            else:
                pdt=self.Tensor_matrix(pdt,operator)
        self.unitary = self.mat_mat_mul(pdt,self.unitary)
        
        return 0   
    
    # 2-qubit cx gate
    
    def cx(self, control, target):
        cx_mat=self.createMatrix(2**self.no_of_qubits)
        for i in range(2**self.no_of_qubits):
            binary=(self.basis_state(i))[::-1]
            if binary[control]=='1':
                if binary[target]=='0':
                    binary=binary[:target]+'1'+binary[target+1:]
                else:
                    binary=binary[:target]+'0'+binary[target+1:]
            dec= self.binaryToDecimal(int(binary[::-1]))
            cx_mat[i][dec]=1
        self.unitary = self.mat_mat_mul(cx_mat,self.unitary)
        
        return 0
    
    def cz(self, control, target):
        self.h(target)
        self.cx(control, target)
        self.h(target)
        
        return 0
    
    def cr(self,angle,control,target):
        cr_mat=self.createMatrix(2**self.no_of_qubits)
        cr_operator=[[math.cos(angle), -math.sin(angle)],[math.sin(angle),math.cos(angle)]]
        for i in range(2**self.no_of_qubits):
            binary=(self.basis_state(i))[::-1]
            if binary[control]=='1':
                if binary[target]=='0':
                    dec_0= self.binaryToDecimal(int(binary[::-1]))
                    binary=binary[:target]+'1'+binary[target+1:]
                    dec_1= self.binaryToDecimal(int(binary[::-1]))
                    cr_mat[dec_0][i]=cr_operator[0][0]
                    cr_mat[dec_1][i]=cr_operator[1][0]
                else:
                    dec_1 = self.binaryToDecimal(int(binary[::-1]))
                    binary = binary[:target]+'0'+binary[target+1:]
                    dec_0 = self.binaryToDecimal(int(binary[::-1]))
                    cr_mat[dec_0][i]=cr_operator[0][1]
                    cr_mat[dec_1][i]=cr_operator[1][1]
            else:
                dec= self.binaryToDecimal(int(binary[::-1]))
                cr_mat[i][dec]=1
        self.unitary = self.mat_mat_mul(cr_mat,self.unitary)
        
        return 0
    
    def ccx(self,control1,control2,target):
        ccx_mat=self.createMatrix(2**self.no_of_qubits)
        for i in range(2**self.no_of_qubits):
            binary=(self.basis_state(i))[::-1]
            if binary[control1]=='1' and binary[control2]=='1':
                if binary[target]=='0':
                    binary=binary[:target]+'1'+binary[target+1:]
                else:
                    binary=binary[:target]+'0'+binary[target+1:]
            dec= self.binaryToDecimal(int(binary[::-1]))
            ccx_mat[i][dec]=1
        self.unitary = self.mat_mat_mul(ccx_mat,self.unitary)
        
        return 0
            
    # Methods for Simulating the circuit
    
    # Method that returns a single unitary matrix (quantum operator) equivalant to all defined 
    # quantum operators until this point
    
    def read_unitary(self):
        return self.unitary
        
        
    # Method that returns the current quantum state of circuit.  
    
    def read_state(self):
        self.state_vector=self.vector_mat_mul(self.state_vector,self.unitary)
        return self.state_vector
        
    
    # Return the probabilisties of observing the basis states if all qubits 
    # are measured at this moment.
    
    def observing_probabilities(self):
        self.state_vector=self.vector_mat_mul(self.state_vector,self.unitary)
        prob={}
        for i in range(int(math.pow(2,self.no_of_qubits))):
            prob[self.basis[i]]=round((self.state_vector[i][0])**2,3)*100
        # print(prob)
        x=list(prob.keys())
        y=list(prob.values())
        plt.bar(x, y, color ='blue', width = 0.4)
        # giving a title to my graph
        plt.xlabel('basis state')
        plt.ylabel('probability')
        plt.title('Probabilities')

        # function to show the plot
        plt.show()
        
    def execute(self, the_number_of_shots=1024):
        result={}
        self.state_vector=self.vector_mat_mul(self.state_vector,self.unitary)
        for i in range(int(math.pow(2,self.no_of_qubits))):
            if self.state_vector[i][0]!=0:
                result[self.basis[i]]=round((self.state_vector[i][0])**2,3)*the_number_of_shots
        return result
                

