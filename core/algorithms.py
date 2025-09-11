import numpy as np
import cvxpy as cp
from collections import deque

def log_loss_bounds(eps=1e-3):
    lmax = -np.log(eps)          
    lmin = -np.log(1.0 - eps)    
    Delta = lmax - lmin          
    return Delta

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_scalar(y, p_hat_vec, ptilde_vec, loss="log", eps=1e-3):
    """Scalar loss for logging and experts update."""
    if loss == "squared":
        losses_hat = (p_hat_vec - y) ** 2
        losses_tld = (ptilde_vec - y) ** 2
    elif loss == "log":
        ph = np.clip(p_hat_vec, eps, 1 - eps)               
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
        eps: float = 1e-3, 
        c: float = 1.0, 
        num_time_steps: int = 1,
        nonadaptive_eta_c: float = 1.0,
        adaptive: bool = True,

        solver: str | None = None
    ):
        self.d = int(d)
        self.m = int(m)
        self.k = 2 * self.m
        self.c = float(c)
        self.adaptive = adaptive
        self.eta = float(eta)
        self.window_size = int(window_size)
        self.gamma_share = 1.0 / (2.0 * self.window_size) if adaptive else 0.0
        self.gamma_pred = float(gamma_pred)
        self.loss = loss
        self.eps = float(eps)
        self.solver = solver
        
        self.num_time_steps = int(num_time_steps)
        self.nonadaptive_eta_c = float(nonadaptive_eta_c)
        
        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.beta = np.zeros(self.d)
        
        # logs
        self.ma_losses = []
        self.reg_losses = []
        
        # trailing window for adaptive eta_t: sum of E[l_MA^2] over last window_size
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
        # vector of f_i(x) * (y - p)
        return fvals * (y - p)

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
            
        # Use A_vec for the sign decision
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
        lreg_bound = log_loss_bounds(self.eps)
        l_reg = float(np.mean((losses_hat - losses_tilde) / lreg_bound))
       
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
        vec_maonly = (fvals_bm * (y_arr - p_hat_vec)[:, None]).mean(axis=0) # (m,)
        vec = np.asarray(vec_maonly, dtype=float).ravel()      
        l_ma_pair = np.concatenate([vec, -vec])                      # (2*m,)
        self.ma_losses.append(l_ma_pair)
        self.reg_losses.append(l_reg)
        
        if self.adaptive:
            # Adaptive eta_t update: E_q[(l_MA)^2] = sum_j (q_{j,+}+q_{j,-}) * (mean loss_j)^2
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
        else:
            self.eta = self.nonadaptive_eta_c * np.sqrt(np.log(self.k) / self.num_time_steps)

        return {"ma_losses": self.ma_losses[-1], "l_reg": l_reg, "eta": self.eta}


class OnlineMARegret:
    """Online multi-accuracy learner with regret minimization.
    """

    def __init__(
        self,
        d: int,
        m: int,
        eta: float = 0.5,
        window_size: int = 100,
        gamma_pred: float = 0.1,
        loss: str = "log",
        eps: float = 1e-3,
        c: float = 1.0,
        num_time_steps: int = 1,
        nonadaptive_eta_c: float = 1.0,
        adaptive: bool = True,
        solver: str | None = None,
    ):
        self.d = int(d)
        self.m = int(m)
        self.k = 2 * self.m + 1
        self.c = float(c)
        self.adaptive = adaptive
        self.eta = float(eta)
        self.window_size = int(window_size)
        self.gamma_share = 1.0 / (2.0 * self.window_size) if self.adaptive else 0.0
        self.gamma_pred = float(gamma_pred)
        self.loss = loss
        self.eps = float(eps)
        self.solver = solver

        self.num_time_steps = int(num_time_steps)
        self.nonadaptive_eta_c = float(nonadaptive_eta_c)

        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.q_reg = uniform
        self.beta = np.zeros(self.d)

        # logs
        self.ma_losses = []
        self.reg_losses = []

        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_reg^2] over last window_size
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0
        
    def _base_predict(self, x: np.ndarray):
        x = np.asarray(x)
        p = sigmoid(x @ self.beta)
        if self.loss == "log":
            p = np.clip(p, self.eps, 1 - self.eps)
        return p  # scalar for 1D x, vector for 2D x

    def _ma_vector(self, fvals: np.ndarray, p: float, y: float) -> np.ndarray:
        # vector of f_i(x) * (y - p)
        return fvals * (y - p)

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

        # Min-max to get p_hat per sample for this batch
        q_diff = (self.q_ma[:, 0] - self.q_ma[:, 1])  # (m,)
        A_vec = fvals_bm @ q_diff  # (b,)
        p_hat_vec = minmax_hat_p_batch(A_vec, self.q_reg, p_tilde_vec, loss=self.loss, eps=self.eps, solver=self.solver)

        # Base learner update (skip if external p_tilde provided)
        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(x, self.beta, y_arr, loss=self.loss)

        # MW update using batch means
        resid_vec = y_arr - p_hat_vec                             # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus
        losses_hat, losses_tilde = loss_scalar(y_arr, p_hat_vec, p_tilde_vec, loss=self.loss, eps=self.eps)
        lreg_bound = log_loss_bounds(self.eps)
        l_reg = float(np.mean((losses_hat - losses_tilde) / lreg_bound))

        exp_plus = self.eta * np.asarray(l_ma_plus,  dtype=float)
        exp_minus = self.eta * np.asarray(l_ma_minus, dtype=float)
        exp_reg = self.eta * np.asarray(l_reg, dtype=float)

        mx = max(exp_reg.max(initial=-np.inf), exp_plus.max(initial=-np.inf), exp_minus.max(initial=-np.inf))
        e_reg = np.exp(exp_reg - mx)
        e_plus = np.exp(exp_plus - mx)
        e_minus = np.exp(exp_minus - mx)

        w_plus = self.q_ma[:, 0] * e_plus
        w_minus = self.q_ma[:, 1] * e_minus
        w_reg = self.q_reg * e_reg
        Z = w_reg + w_plus.sum() + w_minus.sum()
        qhat_ma_plus = w_plus / Z
        qhat_ma_minus = w_minus / Z
        qhat_reg = w_reg / Z

        share = self.gamma_share / self.k
        self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share
        self.q_reg = (1 - self.gamma_share) * qhat_reg + share

        # Logs
        vec = (fvals_bm * (y_arr - p_hat_vec)[:, None]).mean(axis=0) # (m,)
        vec = np.asarray(vec, dtype=float).ravel()      
        l_ma_pair = np.concatenate([vec, -vec])                      # (2*m,)
        self.ma_losses.append(l_ma_pair)
        self.reg_losses.append(l_reg)

        if self.adaptive:
            # Adaptive eta_t update: E_q[(l_MA)^2] + E_q[(l_reg)^2] = sum_j (q_{j,+}+q_{j,-}) * (mean loss_j)^2 + q_reg * (mean loss_reg)^2
            w_ma = self.q_ma[:, 0] + self.q_ma[:, 1]        # (m,)
            w_reg = self.q_reg
            E_ma_sq  = float(np.dot(w_ma, (l_ma_plus**2)))
            E_reg_sq = float(np.dot(w_reg, (l_reg**2)))
            new_sq = self.c * (E_ma_sq + E_reg_sq)
            if len(self._sq_terms) == self._sq_terms.maxlen:
                self._sq_sum -= self._sq_terms[0]
            self._sq_terms.append(new_sq)
            self._sq_sum += new_sq
            denom = max(self.eps, self._sq_sum)
            numer = np.log(2 * self.k * self.window_size) + 1.0
            self.eta = float(np.sqrt(numer / denom))
        else:
            self.eta = self.nonadaptive_eta_c * np.sqrt(np.log(self.k) / self.num_time_steps)
      
        return {"ma_losses": self.ma_losses[-1], "l_reg": l_reg, "eta": self.eta}
