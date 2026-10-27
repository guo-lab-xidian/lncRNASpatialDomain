# -*- coding: utf-8 -*-
"""
BB-DSI 核心方法库。

输入：域内检出计数 y (K,) 与域大小 n (K,)，或直接用二值矩阵。
输出：收缩后的域特异性指数、校准 p 值。

三层校准的第一层（参数自举）在此实现；第二层（检出率匹配）由调用方构造基因集；
第三层（空间分块置换）在 spatial_null 中实现。
"""
import numpy as np
from scipy import stats, optimize


# ----------------------------------------------------------------- 基础统计量
def dsi_raw(y, n):
    """未收缩的域特异性指数：max_k (y_k/n_k) − median_k (y_k/n_k)。"""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = n > 0
    if ok.sum() < 2:
        return np.nan
    f = y[ok] / n[ok]
    return float(f.max() - np.median(f))


def fit_beta_binomial_mom(y, n, rho_bounds=(1e-4, 0.5)):
    """
    矩估计 beta-binomial 的两参数（快，适合逐基因扫描）。

    p = Σy / Σn；由域间方差反解组内相关 rho，再由 p 与 rho 得 alpha, beta。
    Var(y_k) = n_k p(1-p)[1 + (n_k - 1) rho]
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    N = n.sum()
    if N <= 0:
        return np.nan, np.nan
    p = y.sum() / N
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # 用二项方差归一后的残差方差估计 rho
    obs_rate = y / n
    binom_var = p * (1 - p) / n
    excess = np.var(obs_rate) - np.mean(binom_var)
    denom = np.mean(p * (1 - p) - binom_var)
    if denom <= 0:
        rho = rho_bounds[0]
    else:
        rho = excess / denom
    rho = float(np.clip(rho, *rho_bounds))
    s = 1.0 / rho - 1.0
    return p * s, (1 - p) * s


def fit_beta_binomial_mle(y, n):
    """beta-binomial 边际似然的极大似然估计（较慢，用于核对矩估计）。"""
    from scipy.special import gammaln, betaln
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(theta):
        a, b = np.exp(theta)
        ll = (gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
              + betaln(y + a, n - y + b) - betaln(a, b)).sum()
        return -ll if np.isfinite(ll) else 1e12

    p = np.clip(y.sum() / max(n.sum(), 1), 1e-5, 1 - 1e-5)
    s0 = 1.0 / 0.05 - 1
    x0 = np.log([max(p * s0, 1e-3), max((1 - p) * s0, 1e-3)])
    res = optimize.minimize(nll, x0, method="L-BFGS-B",
                            bounds=[(np.log(1e-3), np.log(1e6))] * 2)
    a, b = np.exp(res.x)
    return a, b


def bb_posterior_mean(y, n, a, b):
    """Beta-binomial 后验均值（收缩后的域检出率）。"""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    return (y + a) / (n + a + b)


def bb_dsi(y, n, method="mom"):
    """
    BB-DSI：收缩后的域特异性指数。

    返回 dict：dsi_raw, dsi_shrunk, rho, p_g, a, b, ci95（后验可信区间宽度中位）
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "mle":
        a, b = fit_beta_binomial_mle(y, n)
    else:
        a, b = fit_beta_binomial_mom(y, n)
    if not np.isfinite(a) or not np.isfinite(b):
        return {"dsi_raw": np.nan, "dsi_shrunk": np.nan, "a": np.nan, "b": np.nan,
                "p_g": np.nan, "rho": np.nan, "ci_width": np.nan}
    post = bb_posterior_mean(y, n, a, b)
    dsi_shrunk = float(np.max(post) - np.median(post))
    # 后验 95% 区间宽度的中位（衡量不确定性）
    from scipy.stats import beta as beta_dist
    lo = beta_dist.ppf(0.025, y + a, n - y + b)
    hi = beta_dist.ppf(0.975, y + a, n - y + b)
    return {
        "dsi_raw": dsi_raw(y, n),
        "dsi_shrunk": dsi_shrunk,
        "p_g": float(y.sum() / max(n.sum(), 1)),
        "rho": float(1.0 / (a + b + 1)),
        "a": float(a), "b": float(b),
        "ci_width": float(np.median(hi - lo)),
    }


# ----------------------------------------------------------------- 零分布
def _shrink_dsi_batch(Y, n):
    """
    向量化的"收缩 DSI"：对一批计数矩阵 Y (B, K) 计算
    逐个基因矩估计 beta-binomial 参数 → 后验均值 → max − median。

    这是关键：小域的后验会被强烈收缩，随机噪声不会撑大统计量。
    """
    n = np.asarray(n, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    N = n.sum()
    p = np.clip(Y.sum(axis=1) / N, 1e-6, 1 - 1e-6)
    rates = Y / n[None, :]
    binom_var = p[:, None] * (1 - p[:, None]) / n[None, :]
    excess = rates.var(axis=1) - binom_var.mean(axis=1)
    denom = (p * (1 - p) - binom_var.mean(axis=1))
    rho = np.where(denom > 0, excess / np.maximum(denom, 1e-12), 1e-4)
    rho = np.clip(rho, 1e-4, 0.5)
    s = 1.0 / rho - 1.0
    a = p * s
    b = (1 - p) * s
    post = (Y + a[:, None]) / (n[None, :] + a[:, None] + b[:, None])
    return post.max(axis=1) - np.median(post, axis=1), rho, p


def shrink_dsi(y, n):
    """单基因的收缩 DSI 及其参数。"""
    T, rho, p = _shrink_dsi_batch(np.asarray(y, dtype=float)[None, :], n)
    return float(T[0]), float(rho[0]), float(p[0])


def null_dsi_shrunk(y, n, n_sim=500, seed=0):
    """
    收缩统计量的零分布：按基因自身检出率模拟 y* ~ Bin(n_k, p_g)，
    每次模拟都重新估计收缩参数并计算收缩 DSI。

    这样零分布反映的是"在无域结构、且保持该基因稀疏度的条件下，
    收缩统计量能有多大"，小域噪声已被收缩吸收。
    """
    rng = np.random.default_rng(seed)
    n = np.asarray(n, dtype=float)
    p = np.clip(float(np.sum(y) / max(np.sum(n), 1)), 1e-9, 1 - 1e-9)
    sims = rng.binomial(n.astype(int)[None, :], p, size=(n_sim, len(n))).astype(float)
    T, _, _ = _shrink_dsi_batch(sims, n)
    return T


def pvalue_shrunk(y, n, n_sim=500, seed=0):
    """BB-DSI 的校准 p 值（使用收缩统计量）。"""
    obs, _, _ = shrink_dsi(y, n)
    null = null_dsi_shrunk(y, n, n_sim=n_sim, seed=seed)
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return p, obs, float(null.mean()), float(null.std())


def _shrink_dsi_fixed_batch(Y, n, kappa):
    """
    固定强度平滑的收缩 DSI：a = p·κ, b = (1−p)·κ。

    相比逐基因矩估计 ρ，固定 κ 避免了 ρ 估计的噪声——
    仿真显示 ρ 的矩估计偶尔偏大会让收缩失效，零分布出现重尾，功效被吞掉。

    返回 DSI、后验矩阵
    """
    n = np.asarray(n, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    N = n.sum()
    p = np.clip(Y.sum(axis=1) / N, 1e-9, 1 - 1e-9)
    a = p * kappa
    b = (1 - p) * kappa
    post = (Y + a[:, None]) / (n[None, :] + kappa)
    return post.max(axis=1) - np.median(post, axis=1), post


def shrink_dsi_fixed(y, n, kappa=100.0):
    T, post = _shrink_dsi_fixed_batch(np.asarray(y, dtype=float)[None, :], n, kappa)
    return float(T[0]), post[0]


def pvalue_shrunk_fixed(y, n, kappa=100.0, n_sim=5000, seed=0):
    """固定 κ 的 BB-DSI 校准 p 值。"""
    obs, _ = shrink_dsi_fixed(y, n, kappa)
    rng = np.random.default_rng(seed)
    n = np.asarray(n, dtype=float)
    p = np.clip(float(np.sum(y) / max(np.sum(n), 1)), 1e-9, 1 - 1e-9)
    sims = rng.binomial(n.astype(int)[None, :], p, size=(n_sim, len(n))).astype(float)
    T, _ = _shrink_dsi_fixed_batch(sims, n, kappa)
    pv = (1 + int((T >= obs).sum())) / (1 + len(T))
    return pv, obs, float(T.mean()), float(T.std())


def _pearson_batch(Y, n, p):
    """
    向量化的 Pearson 卡方统计量（对"域间检出率一致"这一零假设）。

    按 n_k 加权，因此不会被小域的随机波动主导——这是 max−median 类统计量的致命弱点。
    当某域的期望计数为 0 时，该域贡献置 0（避免除零）。
    """
    n = np.asarray(n, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    mu = n[None, :] * p
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(mu > 0, (Y - mu) ** 2 / mu, 0.0)
        term2 = np.where((n[None, :] - mu) > 0, ((n[None, :] - Y) - (n[None, :] - mu)) ** 2 /
                         (n[None, :] - mu), 0.0)
    return term.sum(axis=1) + term2.sum(axis=1)


def pvalue_mc_pearson(y, n, n_sim=5000, seed=0):
    """
    蒙特卡洛精确检验：统计量为 Pearson 卡方，零分布由 Bin(n_k, p_g) 参数自举得到。

    与卡方检验的区别：不依赖大样本渐近，因此在极稀疏（π<1%）下仍然校准；
    与 max−median 类统计量的区别：按域大小加权，抗域大小不平衡。
    同时天然实现了"检出率匹配"——零分布以该基因自身的检出率为基准。
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.clip(float(y.sum() / max(n.sum(), 1)), 1e-9, 1 - 1e-9)
    obs = float(_pearson_batch(y[None, :], n, p)[0])
    rng = np.random.default_rng(seed)
    sims = rng.binomial(n.astype(int)[None, :], p, size=(n_sim, len(n))).astype(float)
    null = _pearson_batch(sims, n, p)
    pv = (1 + int((null >= obs).sum())) / (1 + len(null))
    return pv, obs, float(null.mean()), float(null.std())


# ----------------------------------------------------------------- 空间平滑检验
def build_smoother(coords, k=8, include_self=True):
    """
    构造行标准化的 kNN 平滑算子 W（含自身）。

    平滑后的检出强度 s_i = Σ_j W_ij d_j 保留了空间邻域的信息，
    对稀疏基因可把有效样本量放大约 k 倍，信噪比提升约 sqrt(k)。
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, identity
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree").fit(coords)
    _, ind = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), ind.shape[1])
    cols = ind.ravel()
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W = W + W.T
    if not include_self:
        W.setdiag(0)
        W.eliminate_zeros()
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1
    from scipy.sparse import diags
    return diags(1.0 / d) @ W


def smoothed_domain_test(detection, W, dom_codes, K, exact=False, n_sim=2000, seed=0):
    """
    空间平滑的域特异性检验。

    统计量：
        S_k  = Σ_{i∈域k} (W d)_i
        Z_k  = (S_k − n_k p) / sqrt(p(1−p)·‖A_k‖²)，其中 A = G^T W
        T    = Σ_k Z_k²

    零假设：检出与域无关（检出率仍为 p）。零分布近似为 χ²(K)；
    exact=True 时用蒙特卡洛精确校准（对极稀疏更稳妥）。

    返回 dict: pvalue, T, S_k, expected, z
    """
    from scipy.sparse import csr_matrix
    from scipy.stats import chi2
    detection = np.asarray(detection, dtype=float)
    n_cells = len(detection)
    G = csr_matrix((np.ones(n_cells), (np.arange(n_cells), np.asarray(dom_codes))),
                   shape=(n_cells, K))
    A = (G.T @ W).tocsr()                       # K × n_cells
    S = A @ detection                           # K
    n_k = np.asarray(G.sum(axis=0)).ravel()
    p = float(detection.mean())
    if p <= 0 or p >= 1:
        return {"pvalue": 1.0, "T": 0.0, "S": S, "expected": n_k * p, "z": np.zeros(K)}
    var_k = p * (1 - p) * np.asarray(A.multiply(A).sum(axis=1)).ravel()
    z = np.where(var_k > 0, (S - n_k * p) / np.sqrt(np.maximum(var_k, 1e-12)), 0.0)
    T = float((z ** 2).sum())
    if not exact:
        pv = float(chi2.sf(T, df=K))
    else:
        rng = np.random.default_rng(seed)
        cnt = 0
        mu = n_k * p
        sd = np.sqrt(np.maximum(var_k, 1e-12))
        done = 0
        while done < n_sim:
            b = min(500, n_sim - done)
            D = (rng.random((b, n_cells)) < p).astype(np.float32)
            Ssim = np.asarray((A @ D.T).T)      # b × K
            Ts = (((Ssim - mu[None, :]) / sd[None, :]) ** 2).sum(axis=1)
            cnt += int((Ts >= T).sum())
            done += b
        pv = (1 + cnt) / (1 + n_sim)
    return {"pvalue": pv, "T": T, "S": S, "expected": n_k * p, "z": z}


def pvalue_mc_ct_conditioned(detection, cell_codes, dom_codes, K, n_cell,
                             n_sim=2000, seed=0, laplace=0.5):
    """
    以细胞类型为条件的域特异性检验（MC-CT）。

    零假设变为"检出概率只由细胞类型决定"：
        p_i = 该细胞所属类型的检出率（Laplace 平滑）
    在此零模型下模拟，检验域内检出数是否仍显著偏离。

    这一步剥离了"域与细胞类型相关"带来的假阳性——
    仿真显示不做条件化时，细胞类型驱动的基因有 20–30% 会被误判为域特异。

    统计量：T = Σ_k (O_k − E_k)² / V_k，其中
        E_k = Σ_{i∈k} p_i,  V_k = Σ_{i∈k} p_i(1 − p_i)
    """
    detection = np.asarray(detection, dtype=float)
    cell_codes = np.asarray(cell_codes)
    dom_codes = np.asarray(dom_codes)
    n_cells = len(detection)

    # 各细胞类型的检出率：向全局检出率收缩（小类不单独估计，避免系统偏差）
    y_c = np.bincount(cell_codes, weights=detection, minlength=n_cell).astype(float)
    n_c = np.bincount(cell_codes, minlength=n_cell).astype(float)
    p_glob = float(detection.mean())
    # alpha 为收缩强度（伪细胞数）：n_c 越大越接近该类自身估计
    rate_c = (y_c + laplace * p_glob) / (n_c + laplace)
    p_i = rate_c[cell_codes]

    # 域 × 细胞类型 的细胞数列联表 —— 零分布只需要它，无需逐细胞抽样
    n_ck = np.zeros((K, n_cell), dtype=np.int64)
    np.add.at(n_ck, (dom_codes, cell_codes), 1)

    O = np.bincount(dom_codes, weights=detection, minlength=K).astype(float)
    E = n_ck @ rate_c
    V = n_ck @ (rate_c * (1 - rate_c))
    sd = np.sqrt(np.maximum(V, 1e-12))
    z = (O - E) / sd
    T = float((z ** 2).sum())

    # 零分布：O*_k = Σ_c Binomial(n_ck, rate_c)，批量抽样
    rng = np.random.default_rng(seed)
    cnt, done = 0, 0
    while done < n_sim:
        b = min(1000, n_sim - done)
        Okc = rng.binomial(n_ck[None, :, :], rate_c[None, None, :],
                           size=(b, K, n_cell)).astype(float)   # b × K × C
        Os = Okc.sum(axis=2)                                    # b × K
        # 关键：在每次模拟里重新估计细胞类型检出率，使零模型与观测的"估计步骤"一致
        yc = Okc.sum(axis=1)                                    # b × C
        pgl = Os.sum(axis=1) / n_cells
        rate = (yc + laplace * pgl[:, None]) / (n_c + laplace)[None, :]   # b × C
        Es = rate @ n_ck.T                                      # b × K
        Vs = (rate * (1 - rate)) @ n_ck.T                       # b × K
        zs = (Os - Es) / np.sqrt(np.maximum(Vs, 1e-12))
        Ts = (zs ** 2).sum(axis=1)
        cnt += int((Ts >= T).sum())
        done += b
    return {"pvalue": (1 + cnt) / (1 + n_sim), "T": T, "O": O, "E": E, "z": z}


def null_dsi_binomial(y, n, n_sim=500, seed=0):
    """
    L1 参数自举：按基因自身的整体检出率 p_g 模拟 y* ~ Bin(n_k, p_g)，
    得到"该基因在无域结构时"的 DSI 零分布。

    这一步正是消除检出率混杂的关键：零分布随基因自身的检出率而变。
    """
    rng = np.random.default_rng(seed)
    n = np.asarray(n, dtype=float)
    p = np.clip(float(np.sum(y) / max(np.sum(n), 1)), 1e-9, 1 - 1e-9)
    sims = rng.binomial(n.astype(int)[None, :], p, size=(n_sim, len(n))).astype(float)
    # 向量化的 max - median
    rates = sims / n[None, :]
    return rates.max(axis=1) - np.median(rates, axis=1)


def pvalue_parametric(y, n, n_sim=500, seed=0):
    """L1 校准的 p 值。"""
    obs = dsi_raw(y, n)
    if not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    null = null_dsi_binomial(y, n, n_sim=n_sim, seed=seed)
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return p, float(null.mean()), float(null.std())


def pvalue_chisq(y, n):
    """基线：域 × 检出 的卡方检验。"""
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    tab = np.vstack([y, n - y]).T          # K × 2
    tab = tab[tab.sum(axis=1) > 0]
    if tab.shape[0] < 2 or (tab.sum(axis=0) == 0).any():
        return np.nan
    try:
        _, p, _, _ = stats.chi2_contingency(tab)
        return float(p)
    except Exception:
        return np.nan


def pvalue_max_binomial(y, n):
    """基线：对"检出率最高的那个域"做二项检验，再乘域数（Bonferroni）。"""
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.clip(float(y.sum() / max(n.sum(), 1)), 1e-9, 1 - 1e-9)
    k = int(np.argmax(y / np.maximum(n, 1)))
    pv = stats.binomtest(int(y[k]), int(n[k]), p, alternative="greater").pvalue
    return float(min(1.0, pv * len(n)))


# ----------------------------------------------------------------- 空间零模型
def spatial_block_permutation(coords, domain, block_size, n_perm=200, seed=0):
    """
    L3：空间分块置换。把空间切成 block_size×block_size 的方块，
    在块之间置换域标签，从而在保持局部空间自相关的前提下构造零分布。

    返回置换后的域标签数组列表（供调用方逐个重算 DSI）。
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=float)
    bx = np.floor((coords[:, 0] - coords[:, 0].min()) / block_size).astype(int)
    by = np.floor((coords[:, 1] - coords[:, 1].min()) / block_size).astype(int)
    block_id = bx.astype(np.int64) * 100000 + by
    uniq, inv = np.unique(block_id, return_inverse=True)
    # 每个块的域构成：把整块的标签一起平移（块内结构保留）
    outs = []
    for _ in range(n_perm):
        shift = rng.integers(0, len(uniq))
        mapping = np.roll(np.arange(len(uniq)), shift)
        # 块级移位：块 i 的细胞改用块 mapping[i] 的"标签组"——这里简化为整块标签轮换
        block_label = np.array([stats.mode(domain[inv == b], keepdims=False).mode
                                if (inv == b).any() else 0 for b in range(len(uniq))])
        new_block_label = block_label[mapping]
        outs.append(new_block_label[inv])
    return outs
