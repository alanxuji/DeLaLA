import timeit
import numpy as np
from numpy.random import RandomState
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import KernelPCA

# from sklearn.gaussian_process.kernels import RBF



class LMCA:
    """
    Linear transformation matrix Omega learning for kernel large margin component analysis.

    Parameters
    ----------
    k : int, optional (default=2)
        Number of neighbors to use as target neighbors for each sample.

    dimension : int, optional (default=0)
        The dimension of low-dimensional subspace, i.e. Omega.shape[1].
        If 0 , dimension = X_train.shape[1].

    init_method : string, optional (default='rand')
        Initialization of the linear transformation Omega. Possible options are
        'kpca', 'rand' and None.

        kpca:
             KLMCA used the transformation computed by sklearn.decomposition.kernelPCA as initial guess.

        rand:
            Initilize Omega by using numpy.random.RandomState().rand()
            seed can be chosen by parameter:seed.

        None:
            Initilize Omega by np.ones((X_train,shape[0], dimension))

    regularization : float, optional, (default=0.5)
        Weight for second error term , i.e. c in paper.

    stepsize : float, optional (default=5.E-5)
        The stepsize of gradient descent.

    stepsize_min : float, optional (default=1E-20)
        The minimum size of stepsize.

    seed : int, optional (default=None)
        A pseudo random number generator object or a seed for it if int.

    max_iter : int, optional (default=1000)
        Maximum number of iterations in the optimization.

    convergence_tol : float, optional (default=1e-5)
        Convergence tolerance for the optimization.

    length_scale : float, optional (default=1)
        The length scale of RBF.

    gamma : float, optional (default=0)
        The parameter of smooth hinge loss.

    verbose : bool, optional (default=False)
        If True, progress messages will be printed.

    nn_active : bool, optional (default=False)
        If True, KLMCA will compute target neighbors for each sample in every iteration.

    length_scale_test : float, optional (default=0.01)
        The length scale of RBF for test set.

    Examples
    --------
    >>> import LMCA_Mine as lm
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ... stratify=y, test_size=0.25)
    >>> lmca = lm.LMCA(k=3, dimension=2, seed=random_state, init_method="kpca", verbose=True
                , max_iter=20, stepsize=3.E-2, nn_active=False
               , gamma=1.5, length_scale=0.1, regularization=0.2)
    >>> lmca.fit(X_train, y_train)
    >>> X_test_transformed = lmca.cpt_K_test(X_test).dot(lmca.Omega)

    References
    ----------
    .. [1] Torresani ,  Kuang-chih Lee
           "Large Margin Component Analysis"
           Advances in Neural Information Processing Systems, 2007.

    .. [2] MartinHjelm/lmnn
           https://github.com/MartinHjelm/lmnn

    """
    def __init__(self, k=2, dimension=0, max_iter=1000, regularization=0.5, verbose=False, balance=None,
                 init_method='rand', L=None, stepsize=5.E-5, stepsize_min=1E-20, maxcache=3.E9,
                 NNmat=None, seed=None, convergence_tol=1E-1, use_validation=False, nn_weights=None, length_scale=1,
                 gamma=0, nn_active=False, length_scale_test=0.01, **kwargs):

        # super(LMCA, self).__init__(maxcache, seed, verbose, **kwargs)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        # print(self.device)
        # self.L = None
        self.length_scale_test = length_scale_test
        self.K = None  # K为高维核矩阵
        self.Omega = None  # Omega为线性组合系数矩阵，用于更新梯度
        self.k = k
        self.reg = regularization  # Weight for second error term 即论文中的c
        self.gamma = gamma  # smooth hinge 的系数 gamma越大，smooth hinge越接近standard hinge
        self.D = dimension  # 高维目标维度
        self.seed = seed
        self._verbose = verbose
        self.length_scale = length_scale  # The length scale of the kernel.
        self.nn_active = nn_active  # 是否要在迭代中更新目标邻居


        if nn_weights is None:
            self._nn_weights = np.ones(self.k)
        else:
            self._nn_weights = nn_weights

        self._stepsize = stepsize
        self._stepsize_min = stepsize_min

        self._balance = balance
        self._init_method = init_method

        self.NNmat = NNmat

        # Convergence criterias
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.validation = use_validation

        self.num_pts = 0
        self.unique_labels = None
        self.d1 = None
        self._label_idxs = []
        self._non_label_idxs = []  # List of list of indices not belonging to one data class
        self._active_consts = {}
        self.sijl = {}  # sijl存储损失函数第二项中的中间结果LijNorm - LilNorm + 1

        self.trIdx = None
        self.teIdx = None

    def _process_inputs(self, X, labels, Xte=None, yTe=None):
        print("processing inputs...")
        # ASSIGN DATA VARIABLES AND CONTAINERS·
        self.X = X
        self.num_pts = self.X.shape[0]
        self.Xte = Xte  # 测试集
        self.yTe = yTe
        self.d = self.X.shape[1]  # 输入空间维度
        self.labels = labels.astype(int)
        self.unique_labels, label_inds = np.unique(self.labels, return_inverse=True)
        self._active_consts = [0] * self.num_pts  # Create a list with num_pts of [0] : [0, ..., 0]
        self.sijl = [0] * self.num_pts
        self.K = self.cpt_K(self.X)  # 计算输入实例的核矩阵
        # self._init_cache()

        if self.gamma == 0:
            self.gamma = 1 / self.d

        # CHECK THAT DATA AND PARAMS ARE CORRECT
        assert_fail_str = 'The number of labels {:d} do not equal the number of data points {:d}'
        assert len(self.labels) == self.num_pts, (assert_fail_str.format(
            len(labels), self.num_pts))

        # ENOUGH INSTANCES IN EACH CLASS TO DO K-NN
        NminClass = np.bincount(self.labels).min()  # np.bincount() return count number of each labels
        print("NminClass=",NminClass)
        assert_fail_str = 'Not enough class labels for specified k (smallest class has {:d} instances.)'
        assert NminClass > self.k, (assert_fail_str.format(NminClass))

        if self.validation:
            if self.Xte is None:
                raise ValueError('Using validation Xte needs to be set.')
            if self.yTe is None:
                raise ValueError('Using validation yTe needs to be set.')

        # SET PROJECTION DIM
        if self.D == 0:
            self.D = self.d
        if self.d > self.D:
            print("WARNING: Reducing the projection output dimension!", self.d, self.D)

        # # SET DATA IMBALANCE STRATEGY
        # self.ratios = np.ones(len(self.unique_labels))
        # if self._balance == 'ratios':
        #     self.ratios = self.num_pts / np.bincount(self.labels)
        # elif self._balance == 'upsample':
        #     self._upsample_data()

        # LABEL INDICES LISTS
        print("creating LABEL INDICES LISTS...")
        for idx, label in enumerate(
                self.unique_labels):  # enumerate() return enumerate which include the index and element of the list
            self._label_idxs.append(np.flatnonzero(
                self.labels == label))  # return a ndarray of the flatten index of nonzero element in the list 将同标签索引放入一个列表中
            self._non_label_idxs.append(np.flatnonzero(self.labels != label))  # 将与某标签不同标签的数据点的索引位置存入列表

        # COMPUTE NN MATRIX
        print("COMPUTING NN MATRIX...")
        self.kNNs = len(self._nn_weights)
        if self.NNmat is not None:
            if self.NNmat.shape[0] != self.num_pts and self.NNmat.shape[1] != self.k:
                raise RuntimeError(
                    "Shape of NNmat should be N-by-k but is" + str(self.NNmat.shape))
        else:
            # No NN Mat so set to NN
            self.NNmat = np.zeros((self.num_pts, self.kNNs), dtype=int)
            for cls_idxs in self._label_idxs:  # cls_idxs为某一标签的索引集，找到同标签的最近邻并存入NNmat中
                nbrs = NearestNeighbors(n_neighbors=self.kNNs + 1,
                                        algorithm='ball_tree').fit(self.X[cls_idxs, :])
                distances, NNs = nbrs.kneighbors(self.X[cls_idxs,
                                                 :])  # Returns indices of and distances to the neighbors of each point. 其中distance由近到远排序
                self.NNmat[cls_idxs, :] = cls_idxs[NNs[:, 1:self.kNNs + 1]]  # 将最近邻的索引位置存入NNmat，NNmat中每行对应一个数据点的最近邻

        # IMPOSTOR MATRIX Set init value that all ldxs are inactive  and init sijl
        print("COMPUTING IMPOSTOR MATRIX")
        for idx, label in enumerate(self.labels):
            self._active_consts[idx] = {}
            self.sijl[idx] = {}
            for jdx in self.NNmat[idx, :]:
                self._active_consts[idx][jdx] = {}
                self.sijl[idx][jdx] = {}
                for ldx in self._non_label_idxs[label]:
                    self._active_consts[idx][jdx][ldx] = 0  # _active_consts存放激活了hinge loss的三元组
                    self.sijl[idx][jdx][ldx] = 0


        if self.nn_active:
            print("NNMat will be computed in every iter")
            self.d1 = 0  # d1将在梯度中计算
        else:
            # PRECOMPUTE PART OF THE FIRST ERROR TERM IN THE DERIVATIVE (since it is the same all the time) 仅计算delta中第一项(不乘Omega)
            print("PRECOMPUTE PART OF THE FIRST ERROR TERM IN THE DERIVATIVE...")
            self.d1 = np.zeros((self.num_pts, self.num_pts))
            for idx in range(0, self.num_pts):
                for nn_idx, jdx in enumerate(self.NNmat[idx, :]):  # nn_idx表示某个数据点的第几个最近邻，jdx为该最近邻的索引
                    kij = (self.K[idx, :] - self.K[jdx, :]).reshape(1, -1)  # ki - kj保持为行向量
                    Ei, Ej = np.zeros((self.num_pts, self.num_pts)), np.zeros((self.num_pts, self.num_pts))
                    Ei[idx, :] = kij
                    Ej[jdx, :] = kij
                    self.d1 += Ei - Ej

    def EuclidianDist2(self, X1, X2):
        # Using broadcasting, simpler and faster!
        tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  # 行数不知道，只知道列数为1
        tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
        sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)  # 平方差
        sqdist[sqdist < 0] = 0
        return sqdist.T  # 返回X2与X1中所有样本的平方欧氏距离

    def cpt_RBF(self, X, i):  # 获取某一实例xi的核向量Ki
        Ki = np.exp(self.EuclidianDist2(X, X[i, :].reshape(1, - 1)) * self.length_scale * -1.)
        return Ki

    def cpt_K(self, X, K=None):  # 计算所有实例的核矩阵K
        if K is None:
            K = np.zeros((X.shape[0], X.shape[0]))  # K为(实例数，实例数)的矩阵
        for i in range(0, X.shape[0]):
            K[i, :] = self.cpt_RBF(X, i)
        return K

    def init_Omega(self):
        # Check input
        if any([x is None for x in [self.X, self.labels, self.D]]):
            raise ValueError('X, labels and subdim not set!')

        num_pts = self.X.shape[0]
        d = self.X.shape[1]  # 输入空间原始维度
        subdim = self.D  # 目标维度

        # Setup random state
        prng = RandomState()
        if self.seed is not None:
            prng = RandomState(self.seed)
            if self._verbose:
                print("Setting random seed to", self.seed)

        if self._init_method == "rand":
            # method_str = print('Doing random pre-gen Omega')
            bound = np.sqrt(6. / (self.d + self.D))
            Omega = 1. * bound * prng.rand(self.num_pts, subdim) - bound
        if self._init_method == "kpca":
            transformer = KernelPCA(n_components=subdim, kernel='rbf')
            transformer.fit(self.X)
            Omega = transformer.eigenvectors_
        else:  # 初始化为全1矩阵
            Omega = np.ones((self.num_pts, subdim))

        return Omega

    def loss_fun(self, chg_active=False, verbose=False):
        # print("computing loss...")
        er1 = 0
        er2 = 0
        if self.nn_active:  # 需更新目标邻居时更新目标邻居
            self.update_NNMat()

        # For each point...
        for idx, label in enumerate(self.labels):
            # ...and its k nearest neighbors
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):

                Lij = (self.K[idx, :] - self.K[jdx, :]).dot(self.Omega)
                LijNorm = Lij.dot(Lij.T)
                # LOSS TERM 1 1xD * D*d  ∑ij(𝜂𝑖𝑗 * norm((xi - xj).dot(L)))
                er1 += LijNorm

                for ldx in self._non_label_idxs[label]:
                    Lil = (self.K[idx, :] - self.K[ldx, :]).dot(self.Omega)
                    LilNorm = Lil.dot(Lil.T)
                    self.sijl[idx][jdx][ldx] = LijNorm - LilNorm + 1
                    er2 += self.smooth_hinge(self.sijl[idx][jdx][ldx])

                # Deactivate constraints not active only for final loss
                # if chg_active:
                #     for ldx in np.flatnonzero(impostor_dists > 0.):
                #         self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] = 1
                #
                #     for ldx in np.flatnonzero(impostor_dists < 0.):
                #         self._active_consts[idx][jdx][self._non_label_idxs[label][ldx]] = 0

        er = er1 + self.reg * er2

        if self._verbose and verbose:
            info_str = "Er1: {:.3f}, Er2: {:.3f}, Er1/Er2: {:.3f}"
            print(info_str.format(er1, self.reg * er2, (er1 / (self.reg * er2))))

        return np.asscalar(er)  # Convert an array of size 1 to its scalar equivalent. 输出一个数而不是ndarray

    def d_loss_fun(self, active=False):
        # print("computing d_loss...")
        # 梯度第一项已预运算
        if self.nn_active:
            d1 = np.zeros((self.num_pts, self.num_pts))
        d2 = np.zeros((self.num_pts, self.num_pts))

        # For each point...
        for idx, label in enumerate(self.labels):
            # ...and its k nearest neighbors
            for nn_idx, jdx in enumerate(self.NNmat[idx, :]):
                # if self._nn_weights[nn_idx] < 1.E-20:
                #     continue
                kij = (self.K[idx, :] - self.K[jdx, :]).reshape(1, -1)  # ki - kj保持为行向量
                Eij, Eji = np.zeros((self.num_pts, self.num_pts)), np.zeros((self.num_pts, self.num_pts))
                Eij[idx, :] = kij
                Eji[jdx, :] = kij
                if self.nn_active:
                    d1 += Eij - Eji

                # LOSS TERM 2
                if active:
                    # Check only for active impostors to speed up computations
                    ldxActive = []
                    for ldx in self._non_label_idxs[label]:
                        if self._active_consts[idx][jdx][ldx] == 1:
                            ldxActive.append(ldx)
                    if len(ldxActive) == 0:
                        continue

                    # Lil = self._cpt_Xij_diff(idx, ldxActive).dot(self.L)  # i为标量，j为list，直接返回一个矩阵
                else:  # compute all impostors
                    ldxActive = self._non_label_idxs[label]

                for ldx in ldxActive:
                    kil = (self.K[idx, :] - self.K[ldx, :]).reshape(1, -1)  # ki - kl保持为行向量
                    Eil, Eli = np.zeros((self.num_pts, self.num_pts)), np.zeros((self.num_pts, self.num_pts))
                    Eil[idx, :] = kil
                    Eli[ldx, :] = kil
                    E = Eij - Eji - Eil + Eli
                    d2 += self.d_smooth_hinge(self.sijl[idx][jdx][ldx]) * E

        if self.nn_active:
            delta = 2. * (d1 + self.reg * d2).dot(self.Omega)  # 最后乘上2Omega
        else:
            delta = 2. * (self.d1 + self.reg * d2).dot(self.Omega)  # 最后乘上2Omega

        return delta

    def smooth_hinge(self, x):
        return np.log(1 + np.exp(self.gamma * x)) / self.gamma

    def d_smooth_hinge(self, x):
        return 1 - 1 / (1 + np.exp(self.gamma * x))

    def fit(self, X, labels, Xte=None, yTe=None):
        # Precompute neighbors, constant terms of gradient, etc.
        self._process_inputs(X, labels, Xte=Xte, yTe=yTe)

        # Generate a good starting point for the L matrix
        self.Omega = self.init_Omega()
        # Loss. First run sets active false to find all active and inactive constraints
        self.loss_values = [self.loss_fun(chg_active=True)]
        self.Os = [self.Omega]
        self.validation_scores = [0.]
        ticker = 0

        if self._verbose:
            print("Done preprocessing starting GD.")
        start = timeit.default_timer()

        for ii in range(1, self.max_iter + 2):

            # start2 = timeit.default_timer()
            # GRADIENT DESCENT UPDATE
            # Update is Ω = Ω - learn_rate  * dΩ
            # Dimensions: (nxd) = (nxd) - learn_rate * (nxd)
            self.Omega = self.Omega - self._stepsize * self.d_loss_fun()
            # print("Time",timeit.default_timer() - start2)
            # print(' ')
            self.Os.append(self.Omega)
            # if ii % 10 == 0:
            #     self.loss_values.append(self.loss_fun(
            #         self.L, active=False, chg_active=True))
            # else:
            #     self.loss_values.append(self.loss_fun(
            #         self.L, active=False, chg_active=False))

            self.loss_values.append(self.loss_fun())  # 计算损失以及sijl

            # if self.validation:  # compute the score of validation set using L
            #     self.validation_scores.append(self.compute_validation_error())
            # else:
            #     self.validation_scores.append(-1.)

            if self.loss_values[ii] > self.loss_values[ii - 1]:
                self._stepsize *= 0.5
            else:
                self._stepsize *= 1.01

            # CHECK TERMINATION CONDITIONS
            terminate_bool, termination_str = self._test_for_termination(ii)
            if terminate_bool:
                if self._verbose:
                    print(termination_str)
                break

            # Stepsize reset if steps are reduced too fast
            if self._stepsize < self._stepsize_min and ticker < 10:
                ticker += 1
                self._stepsize = 1E-6

            # PRINT ITERATION INFORMATION
            if self._verbose and ii % 1 == 0:
                # if iter%10 == 0 and iter!=0:
                edlstr = "\r"

                if ii % 1 == 0:
                    # Nacs = self._count_active_constraints()
                    loss_diff = self.loss_values[ii] - self.loss_values[ii - 1]
                    loss_diff_prcnt = 100 * (loss_diff / self.loss_values[ii - 1])
                    # nonZeroCols = np.sum(np.linalg.norm(
                    #     self.L, ord=2, axis=1) > 1.E-3)
                    running_time = timeit.default_timer() - start
                    time_per_iter = running_time / float(ii + 1)
                    info_str = ("\033[92mIter: {a:d}\033[0m # Step size: {c:.3g}, "
                                "Er chng: {d:.3g}, Er chng pcnt: {e:3.3g}, Er: {f:3.3g}"
                                ", Time: {i:3.3g} per iter {j:3.3g}     ")
                    # info_str = ("\033[92mIter: {a:d}\033[0m # acs: {b:d}, Step size: {c:.3g}, "
                    #             "Er chng: {d:.3g}, Er chng pcnt: {e:3.3g}, Er: {f:3.3g}, ValEr: {f2:3.3g}, "
                    #             "L-cols: {g:d}/{h:d}, Time: {i:3.3g} per iter {j:3.3g}     ")

                    print(info_str.format(a=ii, c=self._stepsize, d=loss_diff,
                                          e=loss_diff_prcnt, f=self.loss_values[ii], i=running_time,
                                          j=time_per_iter), end=edlstr)
                    # print(info_str.format(a=ii, c=self._stepsize, d=loss_diff,
                    #                       e=loss_diff_prcnt, f=self.loss_values[ii], f2=self.validation_scores[
                    #         ii], i=running_time,
                    #                       j=time_per_iter), end=edlstr)

                    # if ii % 100 == 0 and ii != 0:
                    print('')

        # Go back to when the validation accuracy was best
        if self.validation and ii > 500:
            for i in np.arange(ii, 10, -1):
                if self.validation_scores[i] > self.validation_scores[i - 1]:
                    self.Omega = self.Os[i]
                    break

        # PRINT END INFORMATION
        if self._verbose:
            print(' ')

            lou_kNN, w = self.leave_one_out_average_score(self.X, self.labels, False)
            lou_LMCA, w = self.leave_one_out_average_score(self.X, self.labels)
            print(("Accuracy On Training Set: kNN {:.3f}, LMCA: {:.3f}").format(
                lou_kNN, lou_LMCA))
            if self.Xte is not None:
                K_test = self.cpt_K_test(self.Xte)
                X_test = K_test.dot(self.Omega)
                lou_test, w = self.leave_one_out_average_score(X_test, self.yTe, transform_x=False)
                print(("Accuracy On Test Set: LMCA: {:.3f}").format(lou_test))
            if self.validation:
                print("kNN Validation Accuracy:", self.cpt_knn_score(X, labels, Xte, yTe, k=self.k))
                print("LMCA Validation Accuracy:", self.compute_validation_error())
            running_time = timeit.default_timer() - start
            time_per_iter = running_time / float(ii + 1)
            info_str = "Converged in {:d} iterations and time {:.3f} per iter {:.3f} and Error: {:.3f}"
            print(info_str.format(ii, running_time,
                                  time_per_iter, self.loss_values[ii]))

            # nonZeroCols = np.sum(np.linalg.norm(
            #     self.L.T, ord=2, axis=0) > 1.E-3)
            # print("L Nonzero cols {:d}/{:d}".format(nonZeroCols, self.D))
            # print("L:", self.L)
            print(' ')

        # DELETE STORED MATRIX PRODUCT VALUES GARBAGE COLLECTION TAKES TOO MUCH TIME...
        # self._rm_Xij_dics()

    def _test_for_termination(self, ii=3):
        '''Checks for min step size, convergence of the loss function, max iterations'''

        # Convergence tolerance
        if ii > 10 and np.all(np.abs((100 * (
                self.loss_values[-1] - np.array(self.loss_values[-4:-1])) / self.loss_values[-1])) <= self.convergence_tol):
            # self._stepsize *= 1E-3
            if ii > 10 and np.all(np.abs((100 * (
                    self.loss_values[-1] - np.array(self.loss_values[-10:-1])) / self.loss_values[-1])) <= self.convergence_tol):
                termination_str_nolosschange = "\n\033[92mStopping\033[0m since loss change prcnt is below threshold value."
                return True, termination_str_nolosschange

        # Step size
        if (self._stepsize < self._stepsize_min):
            termination_str_minstepsize = "\n\033[92mStopping\033[0m since step size smaller than min step size."
            return True, termination_str_minstepsize

        # Max iterations
        if ii > self.max_iter:
            termination_str_maxiter = "\n\033[92mStopping\033[0m since maximum number of iterations reached before convergence."
            return True, termination_str_maxiter

        # Validation set goes negative
        if self.validation and ii > 100:
            if np.all(self.validation_scores[-5:] - np.array(self.validation_scores[-6:-1]) <= 0.):
                termination_str_validation = "\n\033[92mStopping\033[0m since validation scores have a negative derivative for 10 steps."
                # print(termination_str_validation)
                return True, termination_str_validation

        # Passed all, so continue
        return False, "Passed termination tests."

    def transform(self):
        return self.K.dot(self.Omega)

    def leave_one_out_average_score(self, X, labels, transform_x=True):
        if transform_x:
            X = self.transform()

        num_pts = X.shape[0]
        idxs = np.arange(0, num_pts)
        scores = 0.
        for te_idx in idxs:
            tr_idx = np.flatnonzero(te_idx != idxs)
            score = self.cpt_knn_score(X[tr_idx, :], labels[tr_idx], X[te_idx, :].reshape(
                1, -1), labels[te_idx].reshape(1, -1), k=self.k)
            scores += score
        return scores / num_pts, np.ones(num_pts)

    # NEAREST NEIGHBOURS FUNCTIONS
    def cpt_knn_score(self, xTr, labels_tr, xTe, labels_te, k=3, score_fun=None):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(xTr, labels_tr)
        if score_fun == 'f1':
            labels_pred = knn.predict(xTe)
            return f1_score(labels_te, labels_pred)
        elif score_fun == 'r2':
            labels_pred = knn.predict(xTe)
            return r2_score(labels_te, labels_pred)
        else:
            return knn.score(xTe, labels_te)

    def transform_X(self, X):
        K = self.cpt_K(X)
        return K.dot(self.Omega)

    def cpt_K_test(self, Xte):
        K_test = np.exp(self.EuclidianDist2(self.X, Xte) * self.length_scale_test * -1)
        return K_test
    def LinearKernel(self, X):
        kernelMat = np.matmul(X, X.T)
        return kernelMat

    def update_NNMat(self):
        for cls_idxs in self._label_idxs:  # cls_idxs为某一标签的索引集，找到同标签的最近邻并存入NNmat中
            X = self.K.dot(self.Omega)  # 在投影空间中寻找目标邻居
            nbrs = NearestNeighbors(n_neighbors=self.kNNs + 1,
                                    algorithm='ball_tree').fit(X[cls_idxs, :])
            # Returns indices of and distances to the neighbors of each point. 其中distance由近到远排序
            distances, NNs = nbrs.kneighbors(X[cls_idxs, :])
            self.NNmat[cls_idxs, :] = cls_idxs[NNs[:, 1:self.kNNs + 1]]  # 将最近邻的索引位置存入NNmat，NNmat中每行对应一个数据点的最近邻
            self.update_sijl()

    def update_sijl(self):
        for idx, label in enumerate(self.labels):  # sijl[idx]一定存在
            for jdx in self.NNmat[idx, :]:
                if jdx not in self.sijl[idx]:  # sijl[idx][jdx]可能不存在
                    self.sijl[idx][jdx] = {}
                    for ldx in self._non_label_idxs[label]:  # 若sijl[idx][jdx]不存在，说明其下没有任何值，应初始化为0
                        self.sijl[idx][jdx][ldx] = 0
