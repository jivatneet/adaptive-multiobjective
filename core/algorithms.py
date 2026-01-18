import numpy as np
import cvxpy as cp
from collections import deque
from dataclasses import dataclass
from math import floor, log2

def log_loss_bounds(eps=1e-3):
    lmax = -np.log(eps)          
    lmin = -np.log(1.0 - eps)    
    Delta = lmax - lmin          
    return Delta

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_scalar(y, p_hat_vec, ptilde_vec, loss="squared", eps=1e-3):
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

def minmax_hat_p_batch(A_vec, q_pred, ptilde_vec, loss="squared", eps=1e-3, solver=None):
    """
    Solve    p̂ = argmin_{p∈[0,1]}  max_{y∈{0,1}}  g_y(p)
    where g_y(p) = A_t (y - p) + q_pred (ℓ(p,y) - ℓ(p̃,y)).

    For log-loss we constrain p ∈ [eps, 1-eps] to keep log well-defined.
    """
    # A_vec: (b,), ptilde_vec: (b,)
    A_vec = np.asarray(A_vec, dtype=float)
    ptilde_vec = np.asarray(ptilde_vec, dtype=float)
    assert np.isfinite(q_pred) and 0 <= q_pred <= 1
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
    lpred_bound = log_loss_bounds(eps)
    if loss == "log":
        g0 = cp.multiply(A_vec, (0.0 - p)) + q_pred * (ell0 - c0) / lpred_bound  # (b,)
        g1 = cp.multiply(A_vec, (1.0 - p)) + q_pred * (ell1 - c1) / lpred_bound  # (b,)
    else:
        g0 = cp.multiply(A_vec, (0.0 - p)) + q_pred * (ell0 - c0)  # (b,)
        g1 = cp.multiply(A_vec, (1.0 - p)) + q_pred * (ell1 - c1)  # (b,)

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

def _bucket_index(p: float, n: int) -> int:
    if p <= 0.0:
        return 1
    if p >= 1.0:
        return n
    i = int(np.floor(p * n)) + 1
    return min(max(1, i), n)

class OnlineMA:

    """Online multiaccuracy learner
    """

    def __init__(
        self, 
        d: int, 
        m: int, 
        eta: float = 0.5, 
        window_size: int = 100, 
        gamma_pred: float = 0.1, 
        loss: str = "squared", 
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

        self.eta = self.nonadaptive_eta_c * np.sqrt(np.log(self.k) / (self.c * self.num_time_steps))
        
        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.beta = np.zeros(self.d)
        
        # logs
        self.ma_losses = []
        self.pred_losses = []
        
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
        if self.loss == "log":
            lpred_bound = log_loss_bounds(self.eps)
            l_pred = float(np.mean((losses_hat - losses_tilde) / lpred_bound))
        else:
            l_pred = float(np.mean(losses_hat - losses_tilde))
       
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
        self.pred_losses.append(l_pred)
        
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

        return {"ma_losses": self.ma_losses[-1], "l_pred": l_pred, "eta": self.eta}


class OnlineMAPred:
    """Online multiaccuracy learner with prediction error minimization.
    """

    def __init__(
        self,
        d: int,
        m: int,
        eta: float = 0.5,
        window_size: int = 100,
        gamma_pred: float = 0.1,
        loss: str = "squared",
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

        self.eta = self.nonadaptive_eta_c * np.sqrt(np.log(self.k) / (self.c * self.num_time_steps))

        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.q_pred = uniform
        self.beta = np.zeros(self.d)

        # logs
        self.ma_losses = []
        self.pred_losses = []

        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_pred^2] over last window_size
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
        p_hat_vec = minmax_hat_p_batch(A_vec, self.q_pred, p_tilde_vec, loss=self.loss, eps=self.eps, solver=self.solver)

        # Base learner update (skip if external p_tilde provided)
        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(x, self.beta, y_arr, loss=self.loss)

        # MW update using batch means
        resid_vec = y_arr - p_hat_vec                             # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus
        losses_hat, losses_tilde = loss_scalar(y_arr, p_hat_vec, p_tilde_vec, loss=self.loss, eps=self.eps)
        if self.loss == "log":
            lpred_bound = log_loss_bounds(self.eps)
            l_pred = float(np.mean((losses_hat - losses_tilde) / lpred_bound))
        else:
            l_pred = float(np.mean(losses_hat - losses_tilde))

        exp_plus = self.eta * np.asarray(l_ma_plus,  dtype=float)
        exp_minus = self.eta * np.asarray(l_ma_minus, dtype=float)
        exp_pred = self.eta * np.asarray(l_pred, dtype=float)

        mx = max(exp_pred.max(initial=-np.inf), exp_plus.max(initial=-np.inf), exp_minus.max(initial=-np.inf))
        e_pred = np.exp(exp_pred - mx)
        e_plus = np.exp(exp_plus - mx)
        e_minus = np.exp(exp_minus - mx)

        w_plus = self.q_ma[:, 0] * e_plus
        w_minus = self.q_ma[:, 1] * e_minus
        w_pred = self.q_pred * e_pred
        Z = w_pred + w_plus.sum() + w_minus.sum()
        qhat_ma_plus = w_plus / Z
        qhat_ma_minus = w_minus / Z
        qhat_pred = w_pred / Z

        share = self.gamma_share / self.k
        self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share
        self.q_pred = (1 - self.gamma_share) * qhat_pred + share

        # Logs
        vec = (fvals_bm * (y_arr - p_hat_vec)[:, None]).mean(axis=0) # (m,)
        vec = np.asarray(vec, dtype=float).ravel()      
        l_ma_pair = np.concatenate([vec, -vec])                      # (2*m,)
        self.ma_losses.append(l_ma_pair)
        self.pred_losses.append(l_pred)

        if self.adaptive:
            # Adaptive eta_t update: E_q[(l_MA)^2] + E_q[(l_pred)^2] = sum_j (q_{j,+}+q_{j,-}) * (mean loss_j)^2 + q_pred * (mean loss_pred)^2
            w_ma = self.q_ma[:, 0] + self.q_ma[:, 1]        # (m,)
            w_pred = self.q_pred
            E_ma_sq  = float(np.dot(w_ma, (l_ma_plus**2)))
            E_pred_sq = float(np.dot(w_pred, (l_pred**2)))
            new_sq = self.c * (E_ma_sq + E_pred_sq)
            if len(self._sq_terms) == self._sq_terms.maxlen:
                self._sq_sum -= self._sq_terms[0]
            self._sq_terms.append(new_sq)
            self._sq_sum += new_sq
            denom = max(self.eps, self._sq_sum)
            numer = np.log(2 * self.k * self.window_size) + 1.0
            self.eta = float(np.sqrt(numer / denom))
        
        return {"ma_losses": self.ma_losses[-1], "l_pred": l_pred, "eta": self.eta}

class OnlineMC:
    """
    Online multicalibration learner on G' = S(f) ∪ G (Algorithm 2, Lee et al., 2022).
    - MW over coordinates (i, s, ±), i∈[n], s∈G', σ∈{+,-}
    - For each example t, solve a tiny LP for x_t ∈ Δ(Ar) (distribution over grid)
      min_{x_t} max_{b∈{0,1}} Σ_i C_t^i [ b·mass_i(x_t) - s_i(x_t) ].
    - Then samples a^t ~ row X_t, and MW-updates using realized vector losses.
    """

    def __init__(
        self,
        n: int,                 # learner buckets
        r: int,                 # refinement (grid spacing 1/(r n))
        m: int,                 # number of raw groups
        n_forecaster: int=None, # |S(f)| (defaults to n)
        num_time_steps: int=1,
        eta: float = 0.5,       # initial η (only used as a starting point if you want)
        loss: str="squared",
        seed: int=42,
        solver: str | None = None, 

        # parameters for local adaptivity
        adaptive: bool = False,
        window_size: int = 100,
        c: float = 4.0,
        eps: float = 1e-3,
    ):
        assert n >= 2 and r >= 1 and m >= 0
        self.n, self.r, self.m = int(n), int(r), int(m)
        self.nf = int(n if n_forecaster is None else n_forecaster)
        self.T = int(num_time_steps)
        self.loss = loss
        self.solver = solver
        self.rng = np.random.default_rng(seed)

        # G' = S(f) ∪ G  (first nf slices are the forecaster level sets; then m raw groups)
        self.S_block = self.nf
        self.Gprime_size = self.nf + self.m

        # Learning rate for EG over (i,s,±)
        self.d = 2 * self.n * self.Gprime_size   
        self.k = self.d            
        base_eta = np.sqrt(np.log(self.d) / (4.0 * max(1, self.T)))
        self.eta = float(min(1.0, base_eta))             

        self.adaptive = bool(adaptive)
        self.window_size = int(window_size)
        self.c = float(c)
        self.eps = float(eps)
        self.gamma_share = 1.0 / (2.0 * self.window_size) if self.adaptive else 0.0

        # Trailing window for adaptive η_t: sum of E_q[(l_MC)^2] 
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0

        # MW weights χ over (i,s,+) and (i,s,-)
        uniform = 1.0 / self.d
        self.q_plus  = np.full((self.n, self.Gprime_size), uniform, dtype=float)
        self.q_minus = np.full((self.n, self.Gprime_size), uniform, dtype=float)

        # Grid a_k = k/(r*n), K = r*n + 1
        self.K = int(self.r * self.n) + 1
        self.grid = np.linspace(0.0, 1.0, self.K)
        # bucket for each grid point (0..n-1)
        self._bucket_of_k = np.minimum(
            np.maximum((np.floor(self.grid * self.n)).astype(int), 0),
            self.n - 1
        )

        # Reusable CVXPY LP parameters for speed
        self._x_var = cp.Variable(self.K, nonneg=True)  # distribution over grid
        self._t_var = cp.Variable()                     # epigraph
        self._mass_param = cp.Parameter(self.K)         # Σ_i C_i · 1{a∈B_i}
        self._s_param    = cp.Parameter(self.K)         # Σ_i C_i · a · 1{a∈B_i}

        self._row_constraints = [
            cp.sum(self._x_var) == 1,
            -self._t_var - self._s_param @ self._x_var <= 0,                         # b = 0
            -self._t_var + (self._mass_param - self._s_param) @ self._x_var <= 0,    # b = 1
        ]
        self._row_problem = cp.Problem(cp.Minimize(self._t_var), self._row_constraints)

        # Logs
        self.ma_losses_hist = []
        self.pred_losses_hist = []

    def _sanitize_prob(self, xrow: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(xrow, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        s = float(x.sum())
        if not np.isfinite(s) or s <= eps:
            x = np.zeros_like(x); x[-1] = 1.0
            return x
        x /= s
        # exact normalization to avoid rare “probabilities do not sum to 1”
        x[-1] = 1.0 - float(x[:-1].sum())
        return np.maximum(x, 0.0)

    def _active_mask(self, g_row: np.ndarray, f_value: float) -> np.ndarray:
        mem = np.zeros(self.Gprime_size, dtype=bool)
        d = _bucket_index(float(f_value), self.nf) - 1  # 0..nf-1
        mem[d] = True
        g_row = g_row.astype(bool, copy=False)
        for gi, on in enumerate(g_row):
            if on:
                mem[self.S_block + gi] = True
        return mem

    def _solve_row_dist(self, C_i: np.ndarray) -> np.ndarray:
        """
        Solve the per-row min–max LP and return x ∈ Δ^K.
        mass_coeff[k] = C_i[bucket(k)],   s_coeff[k] = mass_coeff[k] * grid[k].
        """
        mass_coeff = C_i[self._bucket_of_k]          # (K,)
        s_coeff    = mass_coeff * self.grid          # (K,)
        self._mass_param.value = mass_coeff.astype(float)
        self._s_param.value    = s_coeff.astype(float)

        if self.solver is not None:
            self._row_problem.solve(solver=self.solver, warm_start=True, verbose=False)
        else:
            if "MOSEK" in cp.installed_solvers():
                self._row_problem.solve(solver="MOSEK", warm_start=True, ignore_dpp=True, verbose=False)
            elif "ECOS" in cp.installed_solvers():
                self._row_problem.solve(solver="ECOS",  warm_start=True, ignore_dpp=True, verbose=False)
            else:
                self._row_problem.solve(solver="SCS",   warm_start=True, ignore_dpp=True, verbose=False)

        if self._row_problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Row LP failed: {self._row_problem.status}")

        return self._sanitize_prob(self._x_var.value)

    def update(self, x, y_batch, g_batch, p_tilde):
        """
        y_batch : (b,) labels in [0,1]
        g_batch : (b,m) booleans for raw groups G
        p_tilde : (b,) base forecaster outputs in [0,1] (defines S(f) level set)
        """
        y = np.asarray(y_batch, float).reshape(-1)
        b = y.shape[0]

        G = np.asarray(g_batch)
        if G.ndim == 1:
            G = G.reshape(1, -1)
        assert G.shape == (b, self.m), f"g_batch shape {G.shape} != ({b},{self.m})"
        F = np.asarray(p_tilde, float).reshape(b)

        # Active masks for batch 
        mem = np.zeros((b, self.Gprime_size), dtype=bool)
        for t in range(b):
            mem[t] = self._active_mask(G[t], float(np.clip(F[t], 0.0, 1.0)))

        # C_t = Σ_{s active} (q_plus - q_minus)[:, s]
        diff = (self.q_plus - self.q_minus)  # (n, |G'|)
        # Cache per-mask distributions (q is fixed inside this update)
        dist_cache: dict[tuple[int, ...], np.ndarray] = {}

        preds = np.zeros(b, dtype=float)
        for t in range(b):
            key = tuple(np.flatnonzero(mem[t]))
            if key in dist_cache:
                xt = dist_cache[key]
            else:
                C_t = (diff[:, mem[t]]).sum(axis=1)  # (n,)
                xt = self._solve_row_dist(C_t)
                dist_cache[key] = xt

            idx = self.rng.choice(self.K, p=xt)      # sample a^t ~ x_t
            preds[t] = float(self.grid[idx])

        # Realized vector loss (batch mean)
        l_plus = np.zeros_like(self.q_plus)          # (n, |G'|)
        for t in range(b):
            resid = float(y[t] - preds[t])
            if resid == 0.0:
                continue
            i_star = _bucket_index(preds[t], self.n) - 1
            l_plus[i_star, mem[t]] += resid

        l_plus /= float(b)
        l_minus = -l_plus

        # EG update for q_plus, q_minus
        w_plus  = self.q_plus  * np.exp(self.eta * l_plus)
        w_minus = self.q_minus * np.exp(self.eta * l_minus)
        Z = w_plus.sum() + w_minus.sum()
        qhat_plus  = w_plus  / Z
        qhat_minus = w_minus / Z

        share = self.gamma_share / self.k  
        self.q_plus  = (1.0 - self.gamma_share) * qhat_plus  + share
        self.q_minus = (1.0 - self.gamma_share) * qhat_minus + share

        # Metrics over raw groups G only
        vec = (G * (y - preds)[:, None]).mean(axis=0).astype(float)   # (m,)
        ma_losses = np.concatenate([vec, -vec], axis=0)

        if self.loss != "squared":
            raise NotImplementedError("Only Brier (squared) l_pred implemented.")
        l_pred = float(np.mean((preds - y)**2 - (F - y)**2))

        self.ma_losses_hist.append(ma_losses)
        self.pred_losses_hist.append(l_pred)

        if self.adaptive:
            # V_t = E_mc_sq = E_{(i,s,±) ~ q} [ℓ_{i,s,±}^2]
            w = self.q_plus + self.q_minus             # (n, |G'|)
            E_mc_sq = float(np.sum(w * (l_plus ** 2))) 
            new_sq = self.c * E_mc_sq  

            if len(self._sq_terms) == self._sq_terms.maxlen:
                self._sq_sum -= self._sq_terms[0] 
            self._sq_terms.append(new_sq)
            self._sq_sum += new_sq

            denom = max(self.eps, self._sq_sum)
            # k = total experts = 2 n |G'|
            numer = np.log(self.k * 2.0 * self.window_size) + 1.0
            self.eta = float(np.sqrt(numer / denom))

        return {"ma_losses": ma_losses, "l_pred": l_pred, "p_hat": preds, "eta": self.eta}

class OnlineMCAdaptive:
    """
    Explicit T^2-interval experts for Online MC (Adaptive Regret, Lee et al., 2022).
      - Coordinates: (i, s, τ, r, ±) with i∈[n], s∈G', 1≤τ≤r≤T
      - At absolute time t, form C^i by summing (q_plus - q_minus) over all active slices s
        and all intervals (τ,r) with τ ≤ t ≤ r.
      - Solve per-round min–max LP for x_t over grid A_r, sample a^t ~ x_t,
        then MW-update interval weights using realized vector losses.
    """

    def __init__(
        self,
        n: int,                 # learner buckets
        r: int,                 # refinement (grid spacing 1/(r n))
        m: int,                 # number of raw groups
        n_forecaster: int=None, # |S(f)| (defaults to n)
        num_time_steps: int=1, 
        eta: float = 0.5,
        loss: str="squared",
        seed: int=42,
        solver: str | None = None, 
    ):
        assert n >= 2 and r >= 1 and m >= 0
        self.n, self.r, self.m  = int(n), int(r), int(m)
        self.nf                 = int(n if n_forecaster is None else n_forecaster)
        self.T                  = int(num_time_steps)
        self.loss               = loss
        self.solver             = solver
        self.rng                = np.random.default_rng(seed)

        # G' = S(f) ∪ G
        self.S_block      = self.nf
        self.Gprime_size  = self.nf + self.m

        # Learning rate for MW over (i,s,τ,r,±)
        logF = np.log(self.T * (self.T + 1) / 2.0)  # log(T(T+1)/2)
        logN = np.log(2.0 * self.n * self.Gprime_size) + logF
        self.eta = float(min(1.0, np.sqrt(logN / max(1, self.T)) / np.sqrt(4.0)))

        # Grid a_k = k/(r*n), K = r*n + 1, and bucket index for each grid point
        self.K     = int(self.r*self.n) + 1
        self.grid  = np.linspace(0.0, 1.0, self.K)
        self._bucket_of_k = np.minimum(
            np.maximum((np.floor(self.grid*self.n)).astype(int), 0),
            self.n - 1
        )

        # Interval experts: shape (n, |G'|, T, T) for '+' and '-' (only τ<=r active)
        valid = np.triu(np.ones((self.T, self.T), dtype=bool), k=0)  # (τ, r) with r>=τ
        valid_count = int(valid.sum())
        dcoords = 2 * self.n * self.Gprime_size * valid_count
        uniform = 1.0 / dcoords

        base = uniform * valid.astype(float)  # (T,T), zeros for invalid τ>r
        self.q_plus  = np.zeros((self.n, self.Gprime_size, self.T, self.T), dtype=float)
        self.q_minus = np.zeros((self.n, self.Gprime_size, self.T, self.T), dtype=float)
        # broadcast base into first two dims
        self.q_plus[:,  :, :, :] = base
        self.q_minus[:, :, :, :] = base

        # Reusable per-row LP (min–max over the grid)
        self._x_var       = cp.Variable(self.K, nonneg=True)  # distribution over A_r
        self._t_var       = cp.Variable()                     # epigraph
        self._mass_param  = cp.Parameter(self.K)              # Σ_i C_i · 1{a∈B_i}
        self._s_param     = cp.Parameter(self.K)              # Σ_i C_i · a · 1{a∈B_i}
        self._row_constraints = [
            cp.sum(self._x_var) == 1,
            -self._t_var - self._s_param @ self._x_var <= 0,                         # b=0
            -self._t_var + (self._mass_param - self._s_param) @ self._x_var <= 0,    # b=1
        ]
        self._row_problem = cp.Problem(cp.Minimize(self._t_var), self._row_constraints)

        # Logs + clock
        self.t_global        = 0  # 1-based
        self.ma_losses_hist  = []
        self.pred_losses_hist= []

    def _sanitize_prob(self, xrow: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(xrow, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        s = float(x.sum())
        if not np.isfinite(s) or s <= eps:
            x = np.zeros_like(x); x[-1] = 1.0
            return x
        x /= s
        # exact normalization 
        x[-1] = 1.0 - float(x[:-1].sum())
        return np.maximum(x, 0.0)

    def _active_mask(self, g_row: np.ndarray, f_value: float) -> np.ndarray:
        mem = np.zeros(self.Gprime_size, dtype=bool)
        d = _bucket_index(float(np.clip(f_value, 0.0, 1.0)), self.nf) - 1  # 0..nf-1
        mem[d] = True
        for gi, on in enumerate(g_row.astype(bool, copy=False)):
            if on: mem[self.S_block + gi] = True
        return mem

    def _solve_row_dist(self, C_i: np.ndarray) -> np.ndarray:
        """
        Solve the per-row min–max LP and return x ∈ Δ^K.
        mass_coeff[k] = C_i[bucket(k)],   s_coeff[k] = mass_coeff[k] * grid[k].
        """
        mass_coeff = C_i[self._bucket_of_k]        # (K,)
        s_coeff    = mass_coeff * self.grid        # (K,)
        self._mass_param.value = mass_coeff.astype(float)
        self._s_param.value    = s_coeff.astype(float)

        if self.solver is not None:
            self._row_problem.solve(solver=self.solver, warm_start=True, verbose=False)
        else:
            if "MOSEK" in cp.installed_solvers():
                self._row_problem.solve(solver="MOSEK", warm_start=True, ignore_dpp=True, verbose=False)
            elif "ECOS" in cp.installed_solvers():
                self._row_problem.solve(solver="ECOS",  warm_start=True, ignore_dpp=True, verbose=False)
            else:
                self._row_problem.solve(solver="SCS",   warm_start=True, ignore_dpp=True, verbose=False)

        if self._row_problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Row LP failed: {self._row_problem.status}")

        return self._sanitize_prob(self._x_var.value)

    def update(self, x, y_batch, g_batch, p_tilde):
        """
        y_batch : (b,) labels in [0,1]
        g_batch : (b,m) raw-group indicators
        p_tilde : (b,) base forecaster outputs in [0,1] (defines S(f) level set)
        """
        y = np.asarray(y_batch, float).reshape(-1)
        b = y.shape[0]

        G = np.asarray(g_batch)
        if G.ndim == 1: G = G.reshape(1, -1)
        assert G.shape == (b, self.m), f"g_batch shape {G.shape} != ({b},{self.m})"
        F = np.asarray(p_tilde, float).reshape(b)

        # Active masks for batch
        mem = np.zeros((b, self.Gprime_size), dtype=bool)
        for t in range(b):
            mem[t] = self._active_mask(G[t], float(np.clip(F[t], 0.0, 1.0)))

        # Accumulate 4D losses over (i, s, τ, r) across the batch
        l_plus = np.zeros_like(self.q_plus)   # shape (n, |G'|, T, T)

        preds = np.zeros(b, dtype=float)
        # Absolute time for this entire batch
        t_abs = self.t_global + 1
        tau_slice = slice(0, min(t_abs, self.T))          # τ ∈ {1..t_abs}
        r_slice   = slice(t_abs - 1, self.T)              # r ∈ {t_abs..T}

        for t in range(b):
            # active slices for this example
            cols = np.flatnonzero(mem[t])

            # sum over all active intervals (τ ≤ t_abs ≤ r)
            diff = (self.q_plus[:, cols, tau_slice, r_slice] -
                    self.q_minus[:, cols, tau_slice, r_slice])
            C_i = diff.sum(axis=(1, 2, 3))  # (n,)

            # solve per-row min–max LP and sample a^t
            xt = self._solve_row_dist(C_i)
            idx = self.rng.choice(self.K, p=xt)
            preds[t] = float(self.grid[idx])

            # accumulate interval losses for EG (to every active (τ,r))
            resid = float(y[t] - preds[t])
            if resid != 0.0:
                i_star = _bucket_index(preds[t], self.n) - 1
                l_plus[i_star, cols, tau_slice, r_slice] += resid

        # EG update over all (i, s, τ, r, ±) using batch-mean loss 
        l_plus /= float(b)
        l_minus = -l_plus

        e_plus  = self.q_plus  * np.exp(self.eta * l_plus)
        e_minus = self.q_minus * np.exp(self.eta * l_minus)
        Z = e_plus.sum() + e_minus.sum()
        self.q_plus  = e_plus  / Z
        self.q_minus = e_minus / Z

        # Advance absolute time by one (batch = single round)
        self.t_global += 1

        # Metrics over raw groups G only 
        vec = (G * (y - preds)[:, None]).mean(axis=0).astype(float)   # (m,)
        ma_losses = np.concatenate([vec, -vec], axis=0)

        if self.loss != "squared":
            raise NotImplementedError("Only Brier (squared) l_pred implemented.")
        l_pred = float(np.mean((preds - y)**2 - (F - y)**2))

        self.ma_losses_hist.append(ma_losses)
        self.pred_losses_hist.append(l_pred)
        return {"ma_losses": ma_losses, "l_pred": l_pred, "p_hat": preds, "eta": self.eta}

class OnlineMCSimple:
    """
    Simple multicalibrated + calibeated learner (Algorithm 3, Lee et al., 2022) 
    run on G' = S(f) ∪ G,where S(f) has n level sets and G has m raw groups.

    State: S[i, s] = cumulative signed residual for learner bucket i and slice s∈G'
    Pressures: C^i = sum_{active s} [ exp(η S[i,s]) - exp(-η S[i,s]) ]
    Prediction: two-point randomization between j/n - 1/(r n) and j/n 

    Returns: ma_losses over the original G and l_pred.
    """

    def __init__(
        self,
        n: int,                   # learner buckets
        r: int,                   # refinement inside a bucket (r>=1)
        m: int,       
        n_forecaster: int = None,            
        eta: float = 0.5,        
        loss: str = "squared",
        num_time_steps: int = 1,
        seed: int = 42,
    ):
        assert n >= 2 and r >= 1 and m >= 0
        self.n, self.r, self.m = int(n), int(r), int(m)
        self.nf = int(n if n_forecaster is None else n_forecaster)
        self.loss = loss
        self.T = int(num_time_steps)
        self.rng = np.random.default_rng(seed)

        # G' = S(f) ∪ G 
        # First n slices are the forecaster’s level sets S_1,...,S_n.
        # Next m slices are the raw groups G_1,...,G_m.
        self.S_block = self.nf
        self.Gprime_size = self.nf + self.m

        eta = np.sqrt(np.log(2.0 * self.n * self.Gprime_size) / self.T) / np.sqrt(2.0)
        self.eta = float(min(1.0, eta)) 

        # cumulative residuals per (learner bucket i, slice s)
        self.S = np.zeros((self.n, self.Gprime_size), dtype=float)

    def _active_mask(self, g_row: np.ndarray, f_value: float) -> np.ndarray:
        """Build boolean mask over slices s ∈ G' that are active for this example."""
        mem = np.zeros(self.Gprime_size, dtype=bool)
        # forecaster level-set slice
        d = _bucket_index(float(f_value), self.nf) - 1          # 0..n-1
        mem[d] = True
        # raw groups
        for g_idx, on in enumerate(g_row.astype(bool, copy=False)):
            if on:
                mem[self.S_block + g_idx] = True
        return mem

    def _pressures(self, mem_mask: np.ndarray) -> np.ndarray:
        S_active = self.S[:, mem_mask]                          # (n, # active)
        return (np.exp(self.eta * S_active) - np.exp(-self.eta * S_active)).sum(axis=1)

    def _two_point_predict(self, C: np.ndarray) -> float:
        if np.all(C > 0):  return 1.0
        if np.all(C < 0):  return 0.0

        for j in range(1, self.n):             
            if C[j-1] * C[j] <= 0:
                q = abs(C[j]) / (abs(C[j-1]) + abs(C[j]) + 1e-15)
                left  = j / self.n - 1.0 / (self.r * self.n)  
                right = j / self.n 
                return left if (self.rng.random() < q) else right
        # Fallback 
        return 0.5

    def _update(self, mem_mask: np.ndarray, y: float, p: float):
        i_star = _bucket_index(p, self.n) - 1
        resid = float(y - p)
        if resid != 0.0:
            self.S[i_star, mem_mask] += resid

    def update(self, x, y_batch, g_batch, p_tilde=None):
        """
        y_batch: (b,) labels in [0,1]
        g_batch: (b,m) booleans for raw groups G
        p_tilde: (b,) base forecaster values in [0,1]
        """
        y = np.asarray(y_batch, float).reshape(-1)
        b = y.shape[0]
        G = np.asarray(g_batch)
        if G.ndim == 1:
            G = G.reshape(1, -1)
        assert G.shape == (b, self.m), f"g_batch shape {G.shape} != ({b},{self.m})"
        F = np.asarray(p_tilde, float).reshape(b)

        preds = np.zeros(b, dtype=float)
        for t in range(b):
            mem = self._active_mask(G[t], np.clip(F[t], 0.0, 1.0))
            C = self._pressures(mem)
            p = self._two_point_predict(C)
            preds[t] = p
            self._update(mem, y[t], p)

        vec = (G * (y - preds)[:, None]).mean(axis=0).astype(float)  # (m,)
        ma_losses = np.concatenate([vec, -vec], axis=0)

        l_pred = float(np.mean((preds - y)**2 - (F - y)**2))

        return {"ma_losses": ma_losses, "l_pred": l_pred, "p_hat": preds}

class OnlineMCSimpleMulticalibeating:
    """
    Simple multicalibrated + multicalibeated learner (Algorithm 3, Lee et al., 2022) 
    run on G' = {G_i ∩ S_j(f) : i∈[m], j∈[n]}  ∪  {G_1,...,G_m}
    (optionally also include S(f) itself to add calibeating).

    State: S[i, s] = cumulative signed residual for learner bucket i and slice s∈G'
    Pressures: C^i = sum_{active s} [ exp(η S[i,s]) - exp(-η S[i,s]) ]
    Prediction: two-point randomization between j/n - 1/(r n) and j/n 

    Returns: ma_losses over the original G and l_pred.
    """

    def __init__(
        self,
        n: int,                   
        r: int,                   
        m: int,                   
        n_forecaster: int = None,
        eta: float = 0.5,        
        num_time_steps: int = 1, 
        include_global_S: bool = False, 
        loss: str = "squared",
        seed: int = 42,
    ):
        assert n >= 2 and r >= 1 and m >= 0
        self.n, self.r, self.m = int(n), int(r), int(m)
        self.nf = int(n if n_forecaster is None else n_forecaster)
        self.T = int(num_time_steps)
        self.loss = loss
        self.rng = np.random.default_rng(seed)

        # G' layout:
        # [0 .. m*n-1]            : intersections (G_i ∩ S_j), i-major then j in 0..n-1
        # [m*n .. m*n+m-1]        : raw groups G_i
        # [m*n+m .. m*n+m+n-1]    : (optional) S_j(f) for global calibeating
        self.intersections_size = self.m * self.nf
        self.groups_offset = self.intersections_size
        self.S_offset = self.groups_offset + self.m if include_global_S else None
        self.Gprime_size = self.intersections_size + self.m + (self.nf if include_global_S else 0)

        eta = np.sqrt(np.log(2.0 * self.n * self.Gprime_size) / self.T) / np.sqrt(2.0)
        self.eta = float(min(1.0, eta))

        # cumulative residuals per (learner bucket i, slice s)
        self.S = np.zeros((self.n, self.Gprime_size), dtype=float)

        self.include_global_S = include_global_S

    def _active_mask(self, g_row: np.ndarray, f_value: float) -> np.ndarray:
        """Active slices: for each active G_i, the intersection (G_i ∩ S_d), the raw group G_i, and (optionally) the bare S_d(f)."""
        mem = np.zeros(self.Gprime_size, dtype=bool)

        d = _bucket_index(float(f_value), self.nf) - 1  # 0..n-1 (forecaster level-set)
        # intersections for all active groups
        g_row = g_row.astype(bool, copy=False)
        for i in range(self.m):
            if g_row[i]:
                mem[i * self.n + d] = True                # (G_i ∩ S_d)
                mem[self.groups_offset + i] = True        # raw G_i
        # optional: add S_d for calibeating
        if self.S_offset is not None:
            mem[self.S_offset + d] = True
        return mem

    def _pressures(self, mem_mask: np.ndarray) -> np.ndarray:
        S_active = self.S[:, mem_mask]   # (n, #active)
        return (np.exp(self.eta * S_active) - np.exp(-self.eta * S_active)).sum(axis=1)

    def _two_point_predict(self, C: np.ndarray) -> float:
        if np.all(C > 0):  return 1.0
        if np.all(C < 0):  return 0.0

        for j in range(1, self.n):                
            if C[j-1] * C[j] <= 0:
                q = abs(C[j]) / (abs(C[j-1]) + abs(C[j]) + 1e-15)
                left  = j / self.n - 1.0 / (self.r * self.n)  
                right = j / self.n 
                return left if (self.rng.random() < q) else right
        return 0.5

    def _update(self, mem_mask: np.ndarray, y: float, p: float):
        i_star = _bucket_index(p, self.n) - 1
        self.S[i_star, mem_mask] += float(y - p)

    def update(self, y_batch, g_batch, p_tilde):
        """
        y_batch : (b,) labels in [0,1]
        g_batch : (b,m) booleans for raw groups G
        p_tilde : (b,) base forecaster outputs in [0,1]
        """
        y = np.asarray(y_batch, float).reshape(-1)
        b = y.shape[0]
        G = np.asarray(g_batch)
        if G.ndim == 1:
            G = G.reshape(1, -1)
        assert G.shape == (b, self.m), f"g_batch shape {G.shape} != ({b},{self.m})"
        F = np.asarray(p_tilde, float).reshape(b)

        preds = np.zeros(b, dtype=float)
        for t in range(b):
            mem = self._active_mask(G[t], np.clip(F[t], 0.0, 1.0))
            C = self._pressures(mem)
            p = self._two_point_predict(C)
            preds[t] = p
            self._update(mem, y[t], p)

        vec = (G * (y - preds)[:, None]).mean(axis=0).astype(float)  # (m,)
        ma_losses = np.concatenate([vec, -vec], axis=0)

        if self.loss != "squared":
            raise NotImplementedError("Only squared (Brier) loss implemented.")
        l_pred = float(np.mean((preds - y)**2 - (F - y)**2))

        return {"ma_losses": ma_losses, "l_pred": l_pred, "p_hat": preds}

class OnlineMCAdaptiveEfficient:
    """
    Efficient adaptive online MC (Adaptive Regret, Lee et al., 2022).
      - Coordinates: (i, s, τ, ±) with i∈[n], s∈G', τ∈[1..T]
      - At absolute time t, form C^i by summing (q_plus - q_minus) over all active slices s
        and all intervals starting τ ≤ t (i.e., all intervals [τ, t]).
      - Solve per-round min–max LP for x_t over grid A_r, sample a^t ~ x_t,
        then MW-update interval weights using realized residual.
    """

    def __init__(
        self,
        n: int,                 # learner buckets
        r: int,                 # refinement (grid spacing 1/(r n))
        m: int,                 # number of raw groups
        n_forecaster: int=None, # |S(f)| (defaults to n)
        num_time_steps: int=1, 
        eta: float = 0.5,
        loss: str="squared",
        seed: int=42,
        solver: str | None = None, 
    ):
        assert n >= 2 and r >= 1 and m >= 0
        self.n, self.r, self.m  = int(n), int(r), int(m)
        self.nf                 = int(n if n_forecaster is None else n_forecaster)
        self.T                  = int(num_time_steps)
        self.loss               = loss
        self.solver             = solver
        self.rng                = np.random.default_rng(seed)

        # G' = S(f) ∪ G
        self.S_block      = self.nf
        self.Gprime_size  = self.nf + self.m

        # learning rate EG over (i,s,τ,±)
        logN = np.log(2.0 * self.n * self.Gprime_size) + 2.0 * np.log(self.T)
        self.eta = float(min(1.0, np.sqrt(logN / max(1, self.T)) / np.sqrt(4.0)))

        # Grid a_k = k/(r*n), K = r*n + 1, and bucket index for each grid point
        self.K     = int(self.r*self.n) + 1
        self.grid  = np.linspace(0.0, 1.0, self.K)
        self._bucket_of_k = np.minimum(
            np.maximum((np.floor(self.grid*self.n)).astype(int), 0),
            self.n - 1
        )

        # Interval experts weights: shape (n, |G'|, T) for '+' and '-' 
        dcoords = 2 * self.n * self.Gprime_size * self.T
        uniform = 1.0 / dcoords
        self.q_plus  = np.full((self.n, self.Gprime_size, self.T), uniform, dtype=float)
        self.q_minus = np.full((self.n, self.Gprime_size, self.T), uniform, dtype=float)

        # Reusable per-row LP (min–max over the grid)
        self._x_var       = cp.Variable(self.K, nonneg=True)  # distribution over A_r
        self._t_var       = cp.Variable()                     # epigraph
        self._mass_param  = cp.Parameter(self.K)              # Σ_i C_i · 1{a∈B_i}
        self._s_param     = cp.Parameter(self.K)              # Σ_i C_i · a · 1{a∈B_i}
        self._row_constraints = [
            cp.sum(self._x_var) == 1,
            -self._t_var - self._s_param @ self._x_var <= 0,                         # b=0
            -self._t_var + (self._mass_param - self._s_param) @ self._x_var <= 0,    # b=1
        ]
        self._row_problem = cp.Problem(cp.Minimize(self._t_var), self._row_constraints)

        # Logs + clock
        self.t_global        = 0  # 1-based 
        self.ma_losses_hist  = []
        self.pred_losses_hist= []

    def _sanitize_prob(self, xrow: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(xrow, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.maximum(x, 0.0)
        s = float(x.sum())
        if not np.isfinite(s) or s <= eps:
            x = np.zeros_like(x); x[-1] = 1.0
            return x
        x /= s
        # exact normalization
        x[-1] = 1.0 - float(x[:-1].sum())
        return np.maximum(x, 0.0)

    def _active_mask(self, g_row: np.ndarray, f_value: float) -> np.ndarray:
        mem = np.zeros(self.Gprime_size, dtype=bool)
        d = _bucket_index(float(np.clip(f_value, 0.0, 1.0)), self.nf) - 1  # 0..nf-1
        mem[d] = True
        for gi, on in enumerate(g_row.astype(bool, copy=False)):
            if on: mem[self.S_block + gi] = True
        return mem

    def _solve_row_dist(self, C_i: np.ndarray) -> np.ndarray:
        """
        Solve the per-row min–max LP and return x ∈ Δ^K.
        mass_coeff[k] = C_i[bucket(k)],   s_coeff[k] = mass_coeff[k] * grid[k].
        """
        mass_coeff = C_i[self._bucket_of_k]        # (K,)
        s_coeff    = mass_coeff * self.grid        # (K,)
        self._mass_param.value = mass_coeff.astype(float)
        self._s_param.value    = s_coeff.astype(float)

        if self.solver is not None:
            self._row_problem.solve(solver=self.solver, warm_start=True, verbose=False)
        else:
            if "MOSEK" in cp.installed_solvers():
                self._row_problem.solve(solver="MOSEK", warm_start=True, ignore_dpp=True, verbose=False)
            elif "ECOS" in cp.installed_solvers():
                self._row_problem.solve(solver="ECOS",  warm_start=True, ignore_dpp=True, verbose=False)
            else:
                self._row_problem.solve(solver="SCS",   warm_start=True, ignore_dpp=True, verbose=False)

        if self._row_problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Row LP failed: {self._row_problem.status}")

        return self._sanitize_prob(self._x_var.value)

    def update(self, x, y_batch, g_batch, p_tilde):
        """
        y_batch : (b,) labels in [0,1]
        g_batch : (b,m) raw-group indicators
        p_tilde : (b,) base forecaster outputs in [0,1] (defines S(f) level set)
        """
        y = np.asarray(y_batch, float).reshape(-1)
        b = y.shape[0]

        G = np.asarray(g_batch)
        if G.ndim == 1: G = G.reshape(1, -1)
        assert G.shape == (b, self.m), f"g_batch shape {G.shape} != ({b},{self.m})"
        F = np.asarray(p_tilde, float).reshape(b)

        # Active masks for batch
        mem = np.zeros((b, self.Gprime_size), dtype=bool)
        for t in range(b):
            mem[t] = self._active_mask(G[t], float(np.clip(F[t], 0.0, 1.0)))

        # For EG: accumulate 3D losses over (i, s, τ) across the batch
        l_plus = np.zeros_like(self.q_plus)   # shape (n, |G'|, T)

        self.t_global += 1
        preds = np.zeros(b, dtype=float)
        for t in range(b):
            t_abs = self.t_global
            # active slices
            cols = np.flatnonzero(mem[t])

            # sum over τ<=t_abs for active slices
            tau_slice = slice(0, t_abs)  # [0 .. t_abs-1]
           
            diff = (self.q_plus[:, cols, tau_slice] - self.q_minus[:, cols, tau_slice])
            C_i = diff.sum(axis=(1, 2))  # (n,)

            # solve per-row min–max LP and sample a^t
            xt = self._solve_row_dist(C_i)
            idx = self.rng.choice(self.K, p=xt)
            preds[t] = float(self.grid[idx])

            # accumulate interval losses for EG (broadcast to all τ <= t_abs)
            resid = float(y[t] - preds[t])
            if resid != 0.0:
                i_star = _bucket_index(preds[t], self.n) - 1
                l_plus[i_star, cols, :t_abs] += resid

        # EG update over all (i, s, τ, ±) using batch-mean loss 
        l_plus /= float(b)
        l_minus = -l_plus

        # Multiplicative weights + global normalization
        e_plus  = self.q_plus  * np.exp(self.eta * l_plus)
        e_minus = self.q_minus * np.exp(self.eta * l_minus)
        Z = e_plus.sum() + e_minus.sum()
        self.q_plus  = e_plus  / Z
        self.q_minus = e_minus / Z

        # Metrics over raw groups G only 
        vec = (G * (y - preds)[:, None]).mean(axis=0).astype(float)   # (m,)
        ma_losses = np.concatenate([vec, -vec], axis=0)

        if self.loss != "squared":
            raise NotImplementedError("Only Brier (squared) l_pred implemented.")
        l_pred = float(np.mean((preds - y)**2 - (F - y)**2))

        self.ma_losses_hist.append(ma_losses)
        self.pred_losses_hist.append(l_pred)
        return {"ma_losses": ma_losses, "l_pred": l_pred, "p_hat": preds, "eta": self.eta}

class OnlineMAPredAdaptiveEfficient:
    """Online multiaccuracy learner with prediction error minimization,
    using T^2-interval experts for adaptive regret (similar to OnlineMCAdaptiveEfficient).
    """

    def __init__(
        self,
        d: int,
        m: int,
        eta: float = 0.5,
        window_size: int = 100,
        gamma_pred: float = 0.1,
        loss: str = "squared",
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
        self.window_size = int(window_size)
        self.gamma_pred = float(gamma_pred)
        self.loss = loss
        self.eps = float(eps)
        self.solver = solver

        self.num_time_steps = int(num_time_steps)
        self.nonadaptive_eta_c = float(nonadaptive_eta_c)

        # Learning rate 
        logN = np.log(self.k) + np.log(self.num_time_steps * (self.num_time_steps + 1) / 2.0) 
        self.eta = float(self.nonadaptive_eta_c * np.sqrt(logN / (self.c * max(1, self.num_time_steps))))

        # Interval experts: shape (k, T)
        self.T = self.num_time_steps
        dcoords = self.k * self.T
        uniform = 1.0 / dcoords
        self.q_int = np.full((self.k, self.T), uniform, dtype=float)

        self.q_ma = np.full((self.m, 2), 1.0 / self.k, dtype=float)
        self.q_pred = 1.0 / self.k

        # Base predictor parameters
        self.beta = np.zeros(self.d)

        # logs
        self.ma_losses = []
        self.pred_losses = []

        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_pred^2]
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0

        # global time index (1-based semantics)
        self.t_global = 0

    def _base_predict(self, x: np.ndarray):
        x = np.asarray(x)
        p = sigmoid(x @ self.beta)
        if self.loss == "log":
            p = np.clip(p, self.eps, 1 - self.eps)
        return p  # scalar for 1D x, vector for 2D x

    def _ma_vector(self, fvals: np.ndarray, p: float, y: float) -> np.ndarray:
        # vector of f_i(x) * (y - p)
        return fvals * (y - p)

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray | float,
        fvals: np.ndarray,
        p_tilde: np.ndarray | float | None = None,
    ) -> dict:
        x = np.asarray(x)
        y_arr = np.asarray(y)
        fvals = np.asarray(fvals)

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

        # fvals shape handling: (m,) -> (b,m); (b,m) as is
        if fvals.ndim == 1:
            fvals_bm = np.tile(fvals[None, :], (b, 1))
        else:
            fvals_bm = fvals

        self.t_global += 1
        t_abs = self.t_global  
        w_all = self.q_int[:, :t_abs].sum(axis=1)  # (k,)

        Z = float(w_all.sum())
        if not np.isfinite(Z) or Z <= 0.0:
            w_all[:] = 1.0 / self.k
            Z = 1.0
        mass = w_all / Z  

        # map to MA-plus, MA-minus, pred
        q_ma_plus = mass[:self.m]                # (m,)
        q_ma_minus = mass[self.m:2 * self.m]     # (m,)
        q_pred_curr = float(mass[-1])

        q_ma_curr = np.stack([q_ma_plus, q_ma_minus], axis=1)  # (m, 2)

        q_diff = q_ma_curr[:, 0] - q_ma_curr[:, 1]         # (m,)
        A_vec = fvals_bm @ q_diff                          # (b,)
        p_hat_vec = minmax_hat_p_batch(
            A_vec, q_pred_curr, p_tilde_vec,
            loss=self.loss, eps=self.eps, solver=self.solver
        )

        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(
                x, self.beta, y_arr, loss=self.loss
            )

        resid_vec = y_arr - p_hat_vec                             # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus

        losses_hat, losses_tilde = loss_scalar(
            y_arr, p_hat_vec, p_tilde_vec,
            loss=self.loss, eps=self.eps
        )
        if self.loss == "log":
            lpred_bound = log_loss_bounds(self.eps)
            l_pred = float(np.mean((losses_hat - losses_tilde) / lpred_bound))
        else:
            l_pred = float(np.mean(losses_hat - losses_tilde))

        ell = np.empty(self.k, dtype=float)
        ell[:self.m] = l_ma_plus
        ell[self.m:2 * self.m] = l_ma_minus
        ell[-1] = l_pred

        # accumulate interval losses for EG (broadcast to all τ <= t_abs)
        l_interval = np.zeros_like(self.q_int)      # (k, T)
        l_interval[:, :t_abs] += ell[:, None]       # broadcast to τ = 1..t_abs

        e = self.q_int * np.exp(self.eta * l_interval)
        Zq = float(e.sum())
        if not np.isfinite(Zq) or Zq <= 0.0:
            e[:] = 1.0 / (self.k * self.T)
            Zq = 1.0
        self.q_int = e / Zq

        self.q_ma = q_ma_curr
        self.q_pred = q_pred_curr

        # Multiaccuracy loss vector (2m,) for logging
        vec = (fvals_bm * (y_arr - p_hat_vec)[:, None]).mean(axis=0)  # (m,)
        vec = np.asarray(vec, dtype=float).ravel()
        l_ma_pair = np.concatenate([vec, -vec])                       # (2*m,)

        self.ma_losses.append(l_ma_pair)
        self.pred_losses.append(l_pred)

        return {"ma_losses": self.ma_losses[-1], "l_pred": l_pred, "eta": self.eta}

class OnlineMAPredAblation:
    """Online multiaccuracy learner with prediction error minimization.
    """

    def __init__(
        self,
        d: int,
        m: int,
        eta: float = 0.5,
        window_size: int = 100,
        gamma_pred: float = 0.1,
        loss: str = "squared",
        eps: float = 1e-3,
        c: float = 1.0,
        num_time_steps: int = 1,
        nonadaptive_eta_c: float = 1.0, 
        adaptive: bool = True,
        solver: str | None = None,
        ablation_eta: str = "none", # "interval", "constant", "adaptive"
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
        self.ablation_eta = ablation_eta

        self.num_time_steps = int(num_time_steps)
        self.nonadaptive_eta_c = float(nonadaptive_eta_c)

        if self.ablation_eta == "constant" or self.ablation_eta == "adaptive":
            self.eta = self.nonadaptive_eta_c * np.sqrt(np.log(self.k) / (self.c * self.num_time_steps))
        elif self.ablation_eta == "interval":
            self.eta = np.sqrt((np.log(2 * self.k * self.window_size) + 1.0) / self.window_size)

        uniform = 1.0 / self.k
        self.q_ma = np.full((self.m, 2), uniform)
        self.q_pred = uniform
        self.beta = np.zeros(self.d)

        # logs
        self.ma_losses = []
        self.pred_losses = []

        # trailing window for adaptive eta_t: sum of E[l_MA^2] + E[l_pred^2] over last window_size
        self._sq_terms = deque(maxlen=self.window_size)
        self._sq_sum = 0.0
        
    def _base_predict(self, x: np.ndarray):
        x = np.asarray(x)
        p = sigmoid(x @ self.beta)
        if self.loss == "log":
            p = np.clip(p, self.eps, 1 - self.eps)
        return p  

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
        p_hat_vec = minmax_hat_p_batch(A_vec, self.q_pred, p_tilde_vec, loss=self.loss, eps=self.eps, solver=self.solver)

        # Base learner update (skip if external p_tilde provided)
        if p_tilde is None:
            self.beta = self.beta - self.gamma_pred * grad_beta(x, self.beta, y_arr, loss=self.loss)

        # MW update using batch means
        resid_vec = y_arr - p_hat_vec                             # (b,)
        l_ma_plus = (fvals_bm * resid_vec[:, None]).mean(axis=0)  # (m,)
        l_ma_minus = -l_ma_plus
        losses_hat, losses_tilde = loss_scalar(y_arr, p_hat_vec, p_tilde_vec, loss=self.loss, eps=self.eps)
        if self.loss == "log":
            lpred_bound = log_loss_bounds(self.eps)
            l_pred = float(np.mean((losses_hat - losses_tilde) / lpred_bound))
        else:
            l_pred = float(np.mean(losses_hat - losses_tilde))

        exp_plus = self.eta * np.asarray(l_ma_plus,  dtype=float)
        exp_minus = self.eta * np.asarray(l_ma_minus, dtype=float)
        exp_pred = self.eta * np.asarray(l_pred, dtype=float)

        mx = max(exp_pred.max(initial=-np.inf), exp_plus.max(initial=-np.inf), exp_minus.max(initial=-np.inf))
        e_pred = np.exp(exp_pred - mx)
        e_plus = np.exp(exp_plus - mx)
        e_minus = np.exp(exp_minus - mx)

        w_plus = self.q_ma[:, 0] * e_plus
        w_minus = self.q_ma[:, 1] * e_minus
        w_pred = self.q_pred * e_pred
        Z = w_pred + w_plus.sum() + w_minus.sum()
        qhat_ma_plus = w_plus / Z
        qhat_ma_minus = w_minus / Z
        qhat_pred = w_pred / Z

        share = self.gamma_share / self.k
        self.q_ma[:, 0] = (1 - self.gamma_share) * qhat_ma_plus + share
        self.q_ma[:, 1] = (1 - self.gamma_share) * qhat_ma_minus + share
        self.q_pred = (1 - self.gamma_share) * qhat_pred + share

        # Logs
        vec = (fvals_bm * (y_arr - p_hat_vec)[:, None]).mean(axis=0) # (m,)
        vec = np.asarray(vec, dtype=float).ravel()      
        l_ma_pair = np.concatenate([vec, -vec])                      # (2*m,)
        self.ma_losses.append(l_ma_pair)
        self.pred_losses.append(l_pred)

        if self.ablation_eta == "adaptive":
            # Adaptive eta_t update: E_q[(l_MA)^2] + E_q[(l_pred)^2] = sum_j (q_{j,+}+q_{j,-}) * (mean loss_j)^2 + q_pred * (mean loss_pred)^2
            w_ma = self.q_ma[:, 0] + self.q_ma[:, 1]        # (m,)
            w_pred = self.q_pred
            E_ma_sq  = float(np.dot(w_ma, (l_ma_plus**2)))
            E_pred_sq = float(np.dot(w_pred, (l_pred**2)))
            new_sq = self.c * (E_ma_sq + E_pred_sq)
            if len(self._sq_terms) == self._sq_terms.maxlen:
                self._sq_sum -= self._sq_terms[0]
            self._sq_terms.append(new_sq)
            self._sq_sum += new_sq
            denom = max(self.eps, self._sq_sum)
            numer = np.log(2 * self.k * self.window_size) + 1.0
            self.eta = float(np.sqrt(numer / denom))
        
        return {"ma_losses": self.ma_losses[-1], "l_pred": l_pred, "eta": self.eta}      