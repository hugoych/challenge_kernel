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


class SVM:
    def __init__(self, kernel, C=None, verbose=True, kernel_param=None):
        self.kernel = None
        self.set_kernel(kernel)
        self.kernel_param = kernel_param
        self.C = C
        self.K = None
        self.verbose = verbose
        self.alpha = None
        self.X_sp = None
        self.X = None
        self.Y = None
        self.k = kernel_param.get('k', 3)
        self.X_spectrum = None
        self._spectrum_dict = {e: i for i, e in enumerate(itertools.product(*[range(4)] * self.k))}

    def reset(self, kernel, C=None, verbose=True, kernel_param=None):
        self.kernel = None
        self.set_kernel(kernel)
        self.kernel_param = kernel_param
        self.C = C
        self.K = None
        self.verbose = verbose
        self.alpha = None
        self.X_sp = None
        self.X = None
        self.Y = None
        self.k = kernel_param.get('k', 3)
        self.X_spectrum = None
        self._spectrum_dict = {e: i for i, e in enumerate(itertools.combinations_with_replacement(range(4), self.k))}

    def set_kernel(self, kernel):
        assert (kernel in ['gaussian', 'polynomial', 'spectrum']) or callable(kernel)
        self.kernel = kernel

    def solve(self):
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
        self.alpha = alpha[mask].reshape(-1, 1)
        self.X_sp = self.X[mask]
        if self.kernel == 'spectrum':
            self.X_spectrum = self.X_spectrum[mask]

    def build_K(self, X=None):
        if not callable(self.kernel):
            if self.kernel == 'gaussian':
                return self._gaussian_kernel(X)
            elif self.kernel == 'polynomial':
                return self._polynomial_kernel(X)
            elif self.kernel == 'spectrum':
                return self._spectrum_kernel(X)
        else:
            if X is None:
                n, d = self.X.shape
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1):
                        tmp = self.kernel(self.X[i], self.X[j])
                        K[i][j] = tmp
                        if i != j:
                            K[j][i] = tmp
            else:
                n = X.shape[0]
                nsv = self.X_sp.shape[0]
                K = np.zeros((nsv, n))
                for i in range(nsv):
                    for j in range(n):
                        K[i, j] = self.kernel(self.X_sp[i], X[j])
            return K

    def _gaussian_kernel(self, X=None):
        var = self.kernel_param.get('var', 5.0)
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
        p = self.kernel_param.get('p', 3)
        if X is None:
            XtX = self.X @ self.X.T
        else:
            XtX = self.X_sp @ X.T
        K = np.power(1 + XtX, p)
        return K

    def _to_spectrum(self, X):
        X_spectrum = np.zeros((X.shape[0], X.shape[1] - self.k))
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

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.K = self.build_K()
        self.solve()

    def predict(self, X):
        K = self.build_K(X)
        y = K.T @ self.alpha
        return y

    def score(self, X, Y):
        return np.mean(np.sign(self.predict(X).reshape((-1, 1))) == Y.reshape((-1, 1)))
