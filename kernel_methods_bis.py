import numpy as np
import cvxopt
import time
import itertools


class Hyper:
    def __init__(self, kernels, Cs):
        self.kernels = kernels
        self.Cs = Cs
        self.history = None

    def boost(self, train, validation):
        X_train, Y_train = train
        X_validation, Y_validation = validation
        history = []
        for kernel in self.kernels:
            svm = SVM(kernel[0], kernel_param=kernel[1])
            kernel_name = kernel[0] if not callable(kernel[0]) else kernel[0].__name__
            svm.X = X_train
            svm.Y = Y_train
            svm.build_K()
            for c in self.Cs:
                svm.C = c
                svm.solve()
                acc_train = svm.score(X_train, Y_train)
                acc_test = svm.score(X_validation, Y_validation)
                history.append({'kernel': kernel_name, 'C': c, 'acc_train': acc_train, 'acc_test': acc_test})
                print(history[-1])
        self.history = history
        return history

def LA_kernel(beta,d,e,S):
    def k(x,y):
        n = len(x)
        m = len(y)
        M = np.zeros((n,m))
        X = np.zeros((n,m))
        Y = np.zeros((n,m))
        X2 = np.zeros((n,m))
        Y2 = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                M[i,j] = np.exp(beta*S[x[i]-1,y[j]-1])
                X[i,j] = np.exp(beta*d)*M[i-1,j]+np.exp(beta*e)*X[i-1,j]
                Y[i,j] = np.exp(beta*d)*M[i,j-1]+np.exp(beta*e)*X[i,j-1]
                X2[i,j] = M[i-1,j]+X2[i-1,j]
                Y2[i,j] = M[i,j-1]+X2[i,j-1]+Y2[i,j-1]
        return np.log(1+X2[n-1,m-1]+Y2[n-1,m-1]+M[n-1,m-1])/beta
    return k

class SVM:
    '''
    Class to perform SVM adapted for Hyper
    To add a new kernel:
    If possible create a method _new_kernel(self, X=None) to use matrix product for computation efficiency
    Else create a callable that take into parameter x_i, x_j and returns K(x_i, x_j).
    '''
    def __init__(self, kernels, alpha=[1.],C=None, verbose=True, kernels_param=None):
        '''

        Args:
            kernel (string or callable): Either a string between 'gaussian', 'polynomial', 'spectrum' or a callable
            C (float): Parameter for SVC
            verbose (bool): To debug
            kernel_param (dict): Dict containing the parameters for the kernel
        '''
        self.current_kernel = None
        self.current_param = None
        self.kernels = kernels
        self.kernels_param = kernels_param
        self.C = C
        self.K = None
        self.verbose = verbose
        self.alpha = alpha
        self.X_sp = None
        self.X = None
        self.Y = None
        self.k = None 
        self.l = None
        self.X_spectrum = None
        self.X_mismatch = None
        # spectrum_dict: only for spectrum kernel
        self._spectrum_dict = None 

    def reset(self, kernels, alpha=[1.], C=None, verbose=True, kernels_param=None):
        self.current_kernel = None
        self.current_param = None
        self.kernels = kernels
        self.kernels_param = kernels_param
        self.C = C
        self.K = None
        self.verbose = verbose
        self.alpha = alpha
        self.X_sp = None
        self.X = None
        self.Y = None
        self.k = None
        self.l = None
        self.X_spectrum = None
        self.X_mismatch = None
        self._spectrum_dict = None

    def set_kernel(self, kernel, param):
        '''
        Set a new kernel (doesn't reset the spectrum nor the gramm matrix)
        Args:
            kernel (str or callable):  Either a string between 'gaussian', 'polynomial', 'spectrum' or a callable
        '''
        self.X_spectrum = None
        self.X_mismatch = None
        self._spectrum_dict = None
        self.X_sp = None
        assert(kernel in ['gaussian', 'polynomial', 'spectrum', 'mismatch','substring']) or callable(kernel)
        self.current_kernel = kernel
        self.current_param = param
        #usefull for spectrum and mismatch only
        self.k = param.get('k', 3)
        self.l = param.get('l', 0.6)
        self._spectrum_dict = {e: i for i, e in enumerate(itertools.product(*[range(4)] * self.k))}

    def solve(self):
        '''
        Solve the Convex optimization problem
        '''
        n, d = self.X.shape
        P = cvxopt.matrix(self.K)
        y = self.Y.astype(np.float64)
        q = cvxopt.matrix(-y)
        G = np.zeros((2 * n, n))
        diag = np.diag(y.reshape(-1))
        G[np.arange(2 * n) % 2 == 0] = diag
        G[np.arange(2 * n) % 2 == 1] = -diag
        G = cvxopt.matrix(G)
        h = np.full((2 * n, 1), self.C, dtype=np.float64)
        h[np.arange(2 * n) % 2 == 1] = 0
        h = cvxopt.matrix(h)
        solution = cvxopt.solvers.qp(P, q, G, h, show_progress=self.verbose)
        alpha = np.array(solution['x'])
        mask = np.abs(alpha) > 1e-4
        mask = mask.reshape(-1)

        self.alpha = alpha.reshape(-1, 1)
        self.X_sp = self.X[mask]
        self.Y_sp = self.Y[mask]
        if self.current_kernel == 'spectrum':
            self.X_spectrum = self.X_spectrum[mask]
        if self.current_kernel == 'mismatch':
            self.X_mismatch = self.X_mismatch[mask]



    def build_K(self, X=None):
        '''
        Build the K matrix if X is none it corresponds to the Gramm Matrix
        Args:
            X (np.array or None):

        Returns: (np.array) the K matrix
        '''
        if not callable(self.current_kernel):
            if self.current_kernel == 'gaussian':
                return self._gaussian_kernel(X)
            elif self.current_kernel == 'polynomial':
                return self._polynomial_kernel(X)
            elif self.current_kernel == 'spectrum':
                return self._spectrum_kernel(X)
            elif self.current_kernel == 'mismatch':
                return self._mismatch_kernel(X)
            elif self.current_kernel == 'substring':
                return self._substring_kernel(X)
        else:
            if X is None:
                n, d = self.X.shape
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1):
                        tmp = self.current_kernel(self.X[i], self.X[j])
                        K[i][j] = tmp
                        if i != j:
                            K[j][i] = tmp
                    if i%5==0:
                        print(i)
            else:
                n = X.shape[0]
                nsv = self.X_sp.shape[0]
                K = np.zeros((nsv, n))
                for i in range(nsv):
                    for j in range(n):
                        K[i, j] = self.current_kernel(self.X_sp[i], X[j])
            return K

    def _gaussian_kernel(self, X=None):
        '''
        Compute the K matrix with the Gaussian kernel
        Args:
            X (np.array or None):

        Returns: (np.array) the K matrix

        '''
        var = self.current_param.get('var', 5.0)
        if X is None:
            XtX = self.X @ self.X.T
            diag = np.diag(XtX)
            diag2 = diag
        else:
            XtX = self.X_sp @ X.T
            diag = np.einsum('ij, ij->i', self.X_sp, self.X_sp)
            diag2 = np.einsum('ij, ij->i', X, X)
        K = diag.reshape((-1, 1)) + diag2.reshape((1, -1)) - 2 * XtX
        K = np.exp(-K / (2 * var ** 2))
        return K

    def _polynomial_kernel(self, X=None):
        '''
        Compute the K matrix with the polynomial kernel
        Args:
            X (np.array or None):

        Returns:(np.array) the K matrix

        '''
        p = self.current_param.get('p', 3)
        if X is None:
            XtX = self.X @ self.X.T
        else:
            XtX = self.X_sp @ X.T
        K = np.power(1 + XtX, p)
        return K

    def _to_spectrum(self, X):
        X_spectrum = np.zeros((X.shape[0],len(self._spectrum_dict.keys())))
        for i in range(X.shape[0]):
            for j in range(X.shape[1] - self.k):
                X_spectrum[i, self._spectrum_dict[tuple(X[i][j:j + self.k])]] += 1
        return X_spectrum

    def _spectrum_kernel(self, X=None):
        if self.X_spectrum is None:
            self.X_spectrum = self._to_spectrum(self.X)
        if X is None:
            K = self.X_spectrum @ self.X_spectrum.T
        else:
            X_s = self._to_spectrum(X)
            K = self.X_spectrum @ X_s.T
        return K
    
    def _to_mismatch(self, X):
        X_mismatch = np.zeros((X.shape[0],len(self._spectrum_dict.keys())))
        for i in range(X.shape[0]):
            for j in range(X.shape[1] - self.k):
                seq = list(X[i][j:j + self.k])
                matchs = []
                for l in range(len(seq)):
                    for k in range(4):
                        matchs.append(tuple(seq[:l] + [k] +seq[l+1:]))

                matchs = np.unique(matchs,axis=0)
                for key in matchs:
                    X_mismatch[i, self._spectrum_dict[tuple(key)]] += 1
        return X_mismatch

    def _mismatch_kernel(self, X=None):
        if self.X_mismatch is None:
            
            self.X_mismatch = self._to_mismatch(self.X)

        if X is None:
            
            K = self.X_mismatch @ self.X_mismatch.T
            

        else:
            X_m = self._to_mismatch(X)
            K = self.X_mismatch @ X_m.T
        return K

    def _to_substring(self, X):
        X_substring = np.zeros((X.shape[0],len(self._spectrum_dict.keys())))
        for i in range(X.shape[0]):
            p = 0
            for j in range(X.shape[1]):
                seq = X[i,j:]
                a = X[i,j]
                for h in range(len(seq)):
                    seq2 = seq[h:]
                    b = seq[h]
                    for k in range(len(seq2)):
                        c = seq2[k]
                        #print(self._spectrum_dict.keys())
                        X_substring[i, self._spectrum_dict[tuple([a]+[b]+[c])]] =self.l**(k+h-3)
        return X_substring

    def _substring_kernel(self, X=None):
        if self.X_mismatch is None:
            
            self.X_mismatch = self._to_substring(self.X)

        if X is None:
            
            K = self.X_mismatch @ self.X_mismatch.T
            

        else:
            X_m = self._to_mismatch(X)
            K = self.X_mismatch @ X_m.T
        return K
      

    def fit(self, X, Y):
        '''
        Fit the SVC
        Args:
            X (np.array): input
            Y (np.array): target

        '''
        self.X = X
        self.Y = Y
        n, d = self.X.shape
        self.K = np.zeros((n, n))
        for (kern,al,par) in zip(self.kernels,self.alpha,self.kernels_param):
            self.set_kernel(kern,par)
            self.K += al*self.build_K()
        self.solve()

    def predict(self, X):
        '''
        Make prediction
        Args:
            X (np.array): input

        Returns: (np.array) target

        '''
        K = self.build_K(X)
        y = K.T @ self.alpha
        return y

    def score(self, X, Y):
        '''
        Score the SVC
        Args:
            X (np.array): input
            Y (np.array): target

        Returns: (float) score

        '''
        return np.mean(np.sign(self.predict(X).reshape((-1, 1))) == Y.reshape((-1, 1)))

"""
    def _to_mismatch(self, X):
        X_mismatch = np.zeros((X.shape[0],len(self._spectrum_dict.keys())))
        for i in range(X.shape[0]):
            for j in range(X.shape[1] - self.k):
                seq = list(X[i][j:j + self.k])
                matchs = []
                for l in range(len(seq)):
                    for k in range(4):
                        seq2 = seq[l+1:]
                        for h in range(len(seq2)):
                            for k2 in range(4):
                                matchs.append(tuple(seq[:l] + [k] +seq2[:h] + [k2] +seq2[h+1:]))

                matchs = np.unique(matchs,axis=0)
                for key in matchs:
                    X_mismatch[i, self._spectrum_dict[tuple(key)]] += 1
        return X_mismatch
"""
