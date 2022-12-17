import numpy as np

x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float) #Valeur d'entrée correspondant à la longueur et à la largeur d'une feuille
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype = float) # Donnée de sortie 1 = Rouge/ 0=bleu

x_entrer = x_entrer/np.amax(x_entrer,axis=0)

x = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network (object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize,self.hiddenSize) #Matrice 2*3
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize) #Matrice 3*1

    def forward(self,X): # où X sont les inputs
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, S):
        return 1/(1+np.exp(-S))

    def sigmoidPrime(self,s):
        return s * (1-s)


    def backward(self,X,y,o):
        # On recupere la valeur d'erreur en output (1 seul output)
        self.o_error = y-o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        #On recupere les valeurs d'erreur a la couche de neurone caché
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        #On modifie les valeurs des Synapses
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)

    def predict(self):
        print("Donnée préditte apreès entrainement : ")
        print(" Entrée :  \n "+ str(xPrediction))
        print(" Sortie : \n" +str(self.forward(xPrediction)))

        if (self.forward(xPrediction)< 0.5):
            print("La fleur est bleu ! \n")

        else:
            print("La fleur est Rouge")

NN = Neural_Network()

for i in range (300000):
    # print ("# " + str(i) + "\n")
    # print ( "Valeurs d'entrée : \n " + str(x))
    # print("Sortie Actuelle + \n "+ str(y))
    # print("Sorti preditpar l'IA \n" + str(np.matrix.round(NN.forward(x),2)))
    NN.train(x,y)

print ( "Valeurs d'entrée : \n " + str(x))
print("Sortie Actuelle + \n "+ str(y))
print("Sorti preditpar l'IA \n" + str(np.matrix.round(NN.forward(x),2)))

NN.predict()