import numpy as np 
import cvxopt



def linear_kernel(x1, x2):
    l = max(len(x1), len(x2))
    temp1 = np.zeros(l)
    temp1[:len(x1)] = x1[:]
    temp2 = np.zeros(l)
    temp2[:len(x2)] = x2[:]    
    return np.dot(temp1, temp2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x1, x2, sigma=5.0):
    l = max(len(x1), len(x2))
    temp1 = np.zeros(l)
    temp1[:len(x1)] = x1[:]
    temp2 = np.zeros(l)
    temp2[:len(x2)] = x2[:]    
    return np.exp(-np.linalg.norm(temp1-temp2)**2 / (2 * (sigma ** 2)))


def phi_spectrum(x, k=3):
    phi = np.zeros(4**k)
    len_x = len(x)
    for i in range(len(x)-k+1):
        sub = x[i:i+k]
        ind = 0
        for j in range(k):
            ind += (4**j)*(sub[j]-1)
        phi[ind] += 1
    return phi

def spectrum_kernel(x1,x2):
    phi1 = phi_spectrum(x1)
    phi2 = phi_spectrum(x2)
    return np.dot(phi1,phi2)

class my_svm(object):
    def __init__(self, kernel=linear_kernel,C=None):
        self.kernel = kernel
        self.C = C

    def fit(self,X, y):
        n_samples, n_features = X.shape
        # Compute the Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
               K[i,j] = self.kernel(X[i], X[j])
            if i%50 ==0:
                print(i)
        # construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:      # hard-margin SVM
           G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
           h = cvxopt.matrix(np.zeros(n_samples))
        else:              # soft-margin SVM
           G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
           h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5 # some small threshold
        print(sv)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        return K

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

