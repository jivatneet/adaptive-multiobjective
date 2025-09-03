import numpy as np
import cvxpy as cp
from collections import deque


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_scalar(y, p_hat_vec, ptilde_vec, loss="log", eps=1e-3):
    """Scalar loss for logging and experts update."""
    if loss == "squared":
        losses_hat = (p_hat_vec - y) ** 2
        losses_tld = (ptilde_vec - y) ** 2
    elif loss == "log":
        ph = p_hat_vec                    
        pt = np.clip(ptilde_vec, eps, 1 - eps)  
        losses_hat = -(y * np.log(ph) + (1 - y) * np.log(1 - ph))
        losses_tld = -(y * np.log(pt) + (1 - y) * np.log(1 - pt))
    else:
        raise ValueError("loss must be 'squared' or 'log'.")

    return losses_hat, losses_tld

def grad_beta(x, beta, y, loss="squared"):
    """
    Gradient w.r.t. beta when base predictor is p̃_t = σ(β·x).
    For squared loss: dℓ/dβ = 2(p-y) p(1-p) x
    For log loss:     dℓ/dβ = (p-y) x   (the logistic/xent identity)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        p = sigmoid(x @ beta)
        if loss == "squared":
            return 2.0 * (p - y) * p * (1 - p) * x
        elif loss == "log":
            return (p - y) * x
    else:
        # batch: x shape (b, d), y shape (b,)
        p = sigmoid(x @ beta)
        if loss == "squared":
            grad_samples = (2.0 * (p - y) * p * (1 - p))[:, None] * x
        elif loss == "log":
            grad_samples = (p - y)[:, None] * x
        return grad_samples.mean(axis=0)

def minmax_hat_p_batch(A_vec, q_reg, ptilde_vec, loss="log", eps=1e-3, solver=None):
    """
    Solve    p̂ = argmin_{p∈[0,1]}  max_{y∈{0,1}}  g_y(p)
    where g_y(p) = A_t (y - p) + q_reg (ℓ(p,y) - ℓ(p̃,y)).

    For log-loss we constrain p ∈ [eps, 1-eps] to keep log well-defined.
    """
    # A_vec: (b,), ptilde_vec: (b,)
    A_vec = np.asarray(A_vec, dtype=float)
    ptilde_vec = np.asarray(ptilde_vec, dtype=float)
    assert np.isfinite(q_reg) and 0 <= q_reg <= 1
    assert np.all(np.isfinite(A_vec))
    b = A_vec.size

    p = cp.Variable(b)     # p_i per example
    s = cp.Variable(b)     # epigraph var per example: s_i >= g0(p), s_i >= g1(p)

    if loss == "squared":
        # ℓ(p,0) = p^2, ℓ(p,1) = (p-1)^2
        ell0 = cp.square(p)                         # ℓ(p_i,0)
        ell1 = cp.square(p - 1)                     # ℓ(p_i,1)
        c0 = cp.Constant((ptilde_vec - 0.0)**2)     # constants per i
        c1 = cp.Constant((ptilde_vec - 1.0)**2)
        dom = [p >= 0, p <= 1]
    elif loss == "log":
        # ℓ(p,1) = -log p, ℓ(p,0) = -log(1-p); convex on (0,1)
        # domain constraints to avoid log(0)
        ptc = np.clip(ptilde_vec, eps, 1 - eps)  # only for constants
        c1 = cp.Constant(-np.log(ptc))           # ℓ(p̃_i,1)
        c0 = cp.Constant(-np.log(1 - ptc))       # ℓ(p̃_i,0)
        ell1 = -cp.log(p)                        # ℓ(p_i,1)
        ell0 = -cp.log(1 - p)                    # ℓ(p_i,0)
        dom = [p >= eps, p <= 1 - eps]
    else:
        raise ValueError

    # Elementwise: A_i * (y - p_i)
    g0 = cp.multiply(A_vec, (0.0 - p)) + q_reg * (ell0 - c0)  # shape (b,)
    g1 = cp.multiply(A_vec, (1.0 - p)) + q_reg * (ell1 - c1)  # shape (b,)

    constraints = dom + [s >= g0, s >= g1]
    prob = cp.Problem(cp.Minimize(cp.sum(s)), constraints)
    # Prefer MOSEK if available (fast and stable); otherwise let CVXPY choose defaults
    if solver is not None:
        prob.solve(solver=solver)
    else:
        if "MOSEK" in cp.installed_solvers():
            prob.solve(solver="MOSEK", verbose=False)
        else:
            prob.solve()
     
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"vectorized solve failed: {prob.status}")
    
    return np.asarray(p.value, dtype=float)


class OnlineMA:

    """Online multi-accuracy learner
    """

    def __init__(
        self, 
        d: int, 
        m: int, 
        eta: float = 0.5, 
        window_size: int = 100, 
        gamma_pred: float = 0.1, 
        loss: str = "log", 
        eps: float = 1e-6, 
        c: float = 1.0, 
        solver: str | None = None
    ):
        self.d = int(d)
        self.m = int(m)
        self.k = 2 * self.m
        self.c = float(c)
        self.eta = float(eta)
        self.window_size = int(window_size)
        self.gamma_share = 1.0 / (2.0 * self.window_size)
        self.gamma_pred = float(gamma_pred)
        self.loss = loss
        self.eps = float(eps)
        self.solver = solver
        
        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.beta = np.zeros(self.d)
        
        # logs
        self.ma_l2 = []
        self.ma_linf = []
        self.reg_losses = []
        
        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_reg^2] over last window_size
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0

    # base predictor
    def _base_predict(self, x: np.ndarray):
        x = np.asarray(x)
        p = sigmoid(x @ self.beta)
        if self.loss == "log":
            p = np.clip(p, self.eps, 1 - self.eps)
        return p  # scalar for 1D x, vector for 2D x

    def _ma_vector(self, fvals: np.ndarray, p: float, y: float) -> np.ndarray:
        # vector of f_i(x) * (p - y)
        return fvals * (p - y)

    def update(self, x: np.ndarray, y: np.ndarray | float, fvals: np.ndarray, p_tilde: np.ndarray | float | None = None) -> dict:
        x = np.asarray(x)
        y_arr = np.asarray(y)
        fvals = np.asarray(fvals)
        
        # Determine batch size
        if y_arr.ndim == 0:
            y_arr = np.array([float(y_arr)])
        b = y_arr.shape[0]
        
        # p_tilde vector 
        if p_tilde is None:
            p_tilde_vec = self._base_predict(x)
        else:
            p_tilde_vec = np.asarray(p_tilde)
        if p_tilde_vec.ndim == 0:
            p_tilde_vec = np.full(b, float(p_tilde_vec))
            
        # fvals shape handling: (m,) -> broadcast to (b,m); (b,m) as is
        if fvals.ndim == 1:
            fvals_bm = np.tile(fvals[None, :], (b, 1))
        else:
            fvals_bm = fvals
            
        # Use batch-average A for the sign decision, then compute expected vector.
        q_diff = (self.q_ma[:, 0] - self.q_ma[:, 1])  # (m,)
        A_vec = fvals_bm @ q_diff  # (b,)
        p_hat_vec = np.where(A_vec > 0, 1.0, np.where(A_vec < 0, 0.0, float(0.5)))

        # Base learner update (skip if external p_tilde provided)
        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(x, self.beta, y_arr, loss=self.loss)

        # MW update using batch means
        resid_vec = y_arr - p_hat_vec                             # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus
        losses_hat, losses_tilde = loss_scalar(y_arr, p_hat_vec, p_tilde_vec, loss=self.loss, eps=self.eps)
        l_reg = float(np.mean(losses_hat - losses_tilde))
       
        exp_plus = self.eta * np.asarray(l_ma_plus,  dtype=float)
        exp_minus = self.eta * np.asarray(l_ma_minus, dtype=float)

        mx = max(exp_plus.max(initial=-np.inf), exp_minus.max(initial=-np.inf))
        e_plus  = np.exp(exp_plus - mx)
        e_minus = np.exp(exp_minus - mx)

        w_plus = self.q_ma[:, 0] * e_plus
        w_minus = self.q_ma[:, 1] * e_minus
        Z = w_plus.sum() + w_minus.sum()
        qhat_ma_plus = w_plus / Z
        qhat_ma_minus = w_minus / Z

        share = self.gamma_share / self.k
        self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share

        # Logs
        vec_maonly = (fvals_bm * (y_arr - p_hat_vec )[:, None]).mean(axis=0) # (m,)
        print((fvals_bm * (y_arr - p_hat_vec )[:, None]))
        print((fvals_bm * (y_arr - p_hat_vec )[:, None]).mean(axis=0))
        self.ma_l2.append(float(np.linalg.norm(vec_maonly)))
        self.ma_linf.append(float(np.max(np.abs(vec_maonly))))
        self.reg_losses.append(l_reg)
        
        # Adaptive eta_t update 
        # E_q[(l_MA)^2] = sum_j (q_{j,+}+q_{j,-}) * (mean loss_j)^2
        w_ma = self.q_ma[:, 0] + self.q_ma[:, 1]        # (m,)
        E_ma_sq  = float(np.dot(w_ma, (l_ma_plus**2)))
        new_sq = self.c * E_ma_sq
        if len(self._sq_terms) == self._sq_terms.maxlen:
            self._sq_sum -= self._sq_terms[0]
        self._sq_terms.append(new_sq)
        self._sq_sum += new_sq
        denom = max(self.eps, self._sq_sum)
        numer = np.log(2 * self.k * self.window_size) + 1.0
        self.eta = float(np.sqrt(numer / denom))

        return {"ma_l2": self.ma_l2[-1], "ma_l_infty": self.ma_linf[-1], "l_reg": l_reg, "eta": self.eta}


class OnlineMARegret:
    """Online multi-accuracy learner with regret minimization.

    Usage:
        ma = OnlineMARegret(d=input_dim, m=num_groups, window_size=100, eta=0.5,
                      gamma_pred=0.1, loss="squared")
        metrics = ma.update(x, y, fvals, p_tilde=optional)
    """

    def __init__(
        self,
        d: int,
        m: int,
        eta: float = 0.5,
        window_size: int = 100,
        gamma_pred: float = 0.1,
        loss: str = "log",
        eps: float = 1e-6,
        c: float = 1.0,
        solver: str | None = None,
    ):
        self.d = int(d)
        self.m = int(m)
        self.k = 2 * self.m + 1
        self.c = float(c)
        self.eta = float(eta)
        self.window_size = int(window_size)
        self.gamma_share = 1.0 / (2.0 * self.window_size)
        self.gamma_pred = float(gamma_pred)
        self.loss = loss
        self.eps = float(eps)
        self.solver = solver

        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.q_reg = uniform
        self.beta = np.zeros(self.d)

        # logs
        self.hat_ps = []
        self.tilde_ps = []
        self.L_vals = []
        self.q_hist = []
        self.ma_l2 = []
        self.ma_linf = []
        self.reg_losses = []
        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_reg^2] over last window_size
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0
        # numerical stability for multiplicative weights
        self._exp_clip = 50.0

    def _base_predict(self, x: np.ndarray):
        x = np.asarray(x)
        p = sigmoid(x @ self.beta)
        if self.loss == "log":
            p = np.clip(p, self.eps, 1 - self.eps)
        return p  # scalar for 1D x, vector for 2D x

    def _ma_vector(self, fvals: np.ndarray, p: float, y: float) -> np.ndarray:
        # vector of f_i(x) * (p - y)
        return fvals * (p - y)

    def update(self, x: np.ndarray, y: np.ndarray | float, fvals: np.ndarray, p_tilde: np.ndarray | float | None = None) -> dict:
        x = np.asarray(x)
        y_arr = np.asarray(y)
        fvals = np.asarray(fvals)

        # Determine batch size
        if y_arr.ndim == 0:
            y_arr = np.array([float(y_arr)])
        b = y_arr.shape[0]

        # p_tilde vector and A vector (per-sample)
        if p_tilde is None:
            p_tilde_vec = self._base_predict(x)
        else:
            p_tilde_vec = np.asarray(p_tilde)
        if p_tilde_vec.ndim == 0:
            p_tilde_vec = np.full(b, float(p_tilde_vec))

        # fvals shape handling: (m,) -> broadcast to (b,m); (b,m) as is
        if fvals.ndim == 1:
            fvals_bm = np.tile(fvals[None, :], (b, 1))
        else:
            fvals_bm = fvals

        # A per sample
        q_diff = (self.q_ma[:, 0] - self.q_ma[:, 1])  # (m,)
        A_vec = fvals_bm @ q_diff  # (b,)

        # Min-max to get a single p_hat for this time step/batch
        p_hat, _ = minmax_hat_p_cvxpy(A_vec, self.q_reg, p_tilde_vec, loss=self.loss, eps=self.eps, solver=self.solver)

        # Mixture loss averaged over batch
        losses_hat = np.vectorize(lambda yy: loss_scalar(p_hat, yy, self.loss, self.eps))(y_arr)
        losses_tilde = np.vectorize(lambda yy, pt: loss_scalar(pt, yy, self.loss, self.eps))(y_arr, p_tilde_vec)
        L_t = float(np.mean(A_vec * (y_arr - p_hat) + self.q_reg * (losses_hat - losses_tilde)))

        # Base predictor update (skip if external p_tilde provided)
        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(x, self.beta, y_arr, loss=self.loss)

        # Multiplicative-weights update using batch expectations
        resid_vec = y_arr - p_hat  # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus
        # qtil_ma_plus = self.q_ma[:, 0] * np.exp(self.eta * l_ma_plus)
        # qtil_ma_minus = self.q_ma[:, 1] * np.exp(self.eta * l_ma_minus)
        l_reg = float(np.mean(losses_hat - losses_tilde))

        # Stable log-domain updates with clipping to avoid overflow/underflow
        log_q_plus = np.log(self.q_ma[:, 0] + 1e-300) + np.clip(self.eta * l_ma_plus, -self._exp_clip, self._exp_clip)
        log_q_minus = np.log(self.q_ma[:, 1] + 1e-300) + np.clip(self.eta * l_ma_minus, -self._exp_clip, self._exp_clip)
        log_q_reg = np.log(self.q_reg + 1e-300) + np.clip(self.eta * l_reg, -self._exp_clip, self._exp_clip)

        # new code
        stacked = np.concatenate([log_q_plus, log_q_minus, np.array([log_q_reg])])
        mlog = np.max(stacked)
        unnorm = np.exp(stacked - mlog)
        k = self.m
        qtil_ma_plus = unnorm[:k]
        qtil_ma_minus = unnorm[k:2*k]
        qtil_reg = unnorm[-1]

        Z = qtil_reg + qtil_ma_plus.sum() + qtil_ma_minus.sum()
        qhat_ma_plus = qtil_ma_plus / Z
        qhat_ma_minus = qtil_ma_minus / Z
        qhat_reg = qtil_reg / Z
        # print(f"qhat_ma_plus: {qhat_ma_plus}, qhat_ma_minus: {qhat_ma_minus}, qhat_reg: {qhat_reg}")

        share = self.gamma_share / self.k
        self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share
        self.q_reg = (1 - self.gamma_share) * qhat_reg + share

        # exp_reg  = self.eta * l_reg
        # exp_plus = self.eta * np.asarray(l_ma_plus,  dtype=float)
        # exp_minus= self.eta * np.asarray(l_ma_minus, dtype=float)

        # mx = max(exp_reg, exp_plus.max(initial=-np.inf), exp_minus.max(initial=-np.inf))
        # e_reg   = np.exp(exp_reg  - mx)
        # e_plus  = np.exp(exp_plus - mx)
        # e_minus = np.exp(exp_minus - mx)

        # w_reg = q_reg * e_reg
        # w_plus = q_ma[:, 0] * e_plus
        # w_minus = q_ma[:, 1] * e_minus
        # Z = w_reg + w_plus.sum() + w_minus.sum()
        # qhat_ma_plus = w_plus / Z
        # qhat_ma_minus = w_minus / Z
        # qhat_reg = w_reg / Z

        # share = self.gamma_share / self.k
        # self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        # self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share
        # self.q_reg = (1 - self.gamma_share) * qhat_reg + share


        # Logs
        self.hat_ps.append(p_hat)
        self.tilde_ps.append(float(np.mean(p_tilde_vec)))
        self.L_vals.append(L_t)
        self.q_hist.append((self.q_ma.copy(), self.q_reg))
        self.reg_losses.append(l_reg)

        vec = (fvals_bm * (p_hat - y_arr)[:, None]).mean(axis=0)
        self.ma_l2.append(float(np.linalg.norm(vec)))
        self.ma_linf.append(float(np.max(np.abs(vec))))

        # Adaptive eta_t update
        l_ma_term = float(np.mean(A_vec * resid_vec))
        l_reg_term = float(self.q_reg * l_reg)
        new_sq = self.c * (l_ma_term ** 2 + l_reg_term ** 2)
        if len(self._sq_terms) == self._sq_terms.maxlen:
            self._sq_sum -= self._sq_terms[0]
        self._sq_terms.append(new_sq)
        self._sq_sum += new_sq
        denom = max(self.eps, self._sq_sum)
        numer = np.log(2 * self.k * self.window_size) + 1.0
        self.eta = float(np.sqrt(numer / denom))

        return {"p_hat": p_hat, "p_tilde": float(np.mean(p_tilde_vec)), "L": L_t,
                "ma_l2": self.ma_l2[-1], "ma_l_infty": self.ma_linf[-1],
                "l_reg": l_reg, "eta": self.eta}
