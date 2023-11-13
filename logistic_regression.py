import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(10)

# loading data into predictor matrix and desired value vector
def load_data(filename):
    # reading data
    data = np.loadtxt(filename,delimiter=',',dtype=float)
    # first 5 columns are predictors, last column in true value

    X = data[:,:-1] # predictor matrix
    Y = data[:,-1].reshape(data.shape[0],1) # true value vector
    
    return X,Y

def split_train_test(X,Y,p=0.8):
    
    m,n = X.shape
    
    c = int(m*p)
    
    indices = [i for i in range(m)]
    random.shuffle(indices)
    
    X_train = X[indices[:c],:]
    Y_train = Y[indices[:c]]
    
    X_test = X[indices[c:],:]
    Y_test = Y[indices[c:]]
    
    return X_train, Y_train, X_test, Y_test

def standardization(X_train,X):
    meanX = np.mean(X_train, axis =0)
    stdX = np.std(X_train, axis = 0)
    X = (X - meanX)/stdX
    
    return X

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def likelihood(Y,H):
    return np.prod(H**Y*(1-H)**(1-Y))

def log_likelihood(Y,H):
    return np.sum(Y*np.log(H)+(1-Y)*np.log(1-H))

def train_model(X,Y,lr,max_examples,batch_size,i):
    m,n = X.shape
    data = []
    examples = []
    Theta = np.ones((n,1))
    
    data.append(likelihood(Y,sigmoid(X@Theta)))
    examples.append(0)
    
    start = 0
    end = batch_size
    
    while examples[-1] < max_examples:
        dl = X[start:end,:].T@(Y[start:end,:]-sigmoid(X[start:end,:]@Theta))
        dl /= end - start
    
        Theta = Theta + lr*dl
        
        data.append(likelihood(Y,sigmoid(X@Theta)))
        
        start += batch_size
        end += batch_size
        if end > X.shape[0]:
            end = X.shape[0]
        if start >= X.shape[0]:
            start = 0
            end = batch_size
            
        examples.append(examples[-1] + end - start)
    
    if i == 2:
        plt.figure()
        plt.plot(examples,data)
        plt.xlabel("broj primera")
        plt.ylabel("$L$")
        plt.show()
    return Theta

def train_all(X,Y,lr=0.1,max_examples = 1000000, batch_size = 16):
    Y_new = np.zeros(Y.shape)
    Theta = []
    for i in range(3):
        for j in range(len(Y)):
            if Y[j] == i:
                Y_new[j] = 1
            else:
                Y_new[j] = 0
                
        Theta.append(train_model(X,Y_new,lr,max_examples,batch_size,i))
    
    return Theta

def predict(X,Theta):
    outputs = np.zeros((X.shape[0],len(Theta)))
    for i in range(len(Theta)):
        outputs[:,i] = sigmoid((X@Theta[i]).reshape((X.shape[0],)))
    
    Y_pred = np.argmax(outputs,axis=1)
    return Y_pred
        

X, Y = load_data('multiclass_data.csv')

X_train, Y_train, X_test, Y_test = split_train_test(X,Y)

X_test = standardization(X_train,X_test)
X_train = standardization(X_train,X_train)

Theta = train_all(X_train,Y_train)

Y_pred_train = predict(X_train,Theta)
Y_pred_train = Y_pred_train.reshape(Y_pred_train.shape[0],1)
Y_pred_test = predict(X_test,Theta)
Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0],1)

print("Accuracy on Train: " + str(np.sum(Y_pred_train == Y_train)/len(Y_train)*100)[:7] + " %")
print("Accuracy on Test:  " + str(np.sum(Y_pred_test == Y_test)/len(Y_test)*100)[:7] + " %")



