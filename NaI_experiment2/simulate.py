import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


def klein_nishina_weights(mu, e0_mev):
    mec2 = 0.511
    alpha = e0_mev / mec2
    e_ratio = 1.0 / (1.0 + alpha * (1.0 - mu))
    term = e_ratio + 1.0 / e_ratio - (1.0 - mu * mu)
    w = e_ratio * e_ratio * term
    return w


def _compton_w_max(e0_mev: float) -> float:
    mu_grid = np.linspace(-1.0, 1.0, 4097)
    w_grid = klein_nishina_weights(mu_grid, e0_mev)
    return float(w_grid.max()) * 1.05


def sample_compton_edeps(n: int, e0_mev: float, rng: np.random.Generator, w_max: float) -> np.ndarray:
    mec2 = 0.511
    alpha = e0_mev / mec2

    out = np.empty(n, dtype=np.float32)
    filled = 0
    while filled < n:
        need = n - filled
        k = max(8192, int(need * 4))
        mu = rng.uniform(-1.0, 1.0, size=k).astype(np.float32)
        w = klein_nishina_weights(mu, e0_mev).astype(np.float32)
        u = rng.uniform(0.0, w_max, size=k).astype(np.float32)
        mu_sel = mu[u <= w]
        if mu_sel.size == 0:
            continue
        e_prime = e0_mev / (1.0 + alpha * (1.0 - mu_sel))
        edep = (e0_mev - e_prime).astype(np.float32)
        take = min(need, edep.size)
        out[filled:filled + take] = edep[:take]
        filled += take
    return out


def resolution_sigma(e_mev, fwhm_rel_662=0.07):
    k = fwhm_rel_662 * np.sqrt(0.662)
    fwhm = k * np.sqrt(np.maximum(e_mev, 1e-9))
    sigma = fwhm / 2.355
    return sigma


def simulate_histogram(e0_mev=2.0, n_events=120000, w_full=0.15, w_back=0.05, seed=42, bins=None):
    rng = np.random.default_rng(seed)
    if bins is None:
        bins = np.linspace(0.0, 2.2, 440)
    bins = np.asarray(bins, dtype=float)

    n_full = max(0, int(n_events * w_full))
    n_back = max(0, int(n_events * w_back))
    n_comp = max(0, n_events - n_full - n_back)

    hist = np.zeros(bins.size - 1, dtype=np.int64)

    e_back = e0_mev / (1.0 + 2.0 * e0_mev / 0.511)
    e_min = float(bins[0])
    e_max = float(bins[-1])

    if n_full:
        sigma = float(resolution_sigma(np.asarray([e0_mev], dtype=float))[0])
        e_full = (e0_mev + rng.normal(0.0, sigma, size=n_full)).astype(np.float32)
        e_full = e_full[(e_full >= e_min) & (e_full <= e_max)]
        hist += np.histogram(e_full, bins=bins)[0]

    if n_back:
        sigma = float(resolution_sigma(np.asarray([e_back], dtype=float))[0])
        e_bs = (e_back + rng.normal(0.0, sigma, size=n_back)).astype(np.float32)
        e_bs = e_bs[(e_bs >= e_min) & (e_bs <= e_max)]
        hist += np.histogram(e_bs, bins=bins)[0]

    if n_comp:
        w_max = _compton_w_max(e0_mev)
        chunk = 20000
        n_chunks = int(np.ceil(n_comp / chunk))
        for i in trange(n_chunks, desc="Compton", unit="chunk"):
            m = chunk if i < n_chunks - 1 else (n_comp - chunk * (n_chunks - 1))
            edep = sample_compton_edeps(m, e0_mev, rng, w_max=w_max)
            sig = resolution_sigma(edep.astype(float)).astype(np.float32)
            edep = edep + rng.normal(0.0, sig, size=m).astype(np.float32)
            edep = edep[(edep >= e_min) & (edep <= e_max)]
            hist += np.histogram(edep, bins=bins)[0]

    centers = 0.5 * (bins[:-1] + bins[1:])
    return hist, centers, e_back


def main():
    e0 = 2.0
    bins = np.linspace(0.0, 2.2, 440)
    hist, centers, e_sc_180 = simulate_histogram(e0_mev=e0, n_events=120000, w_full=0.16, w_back=0.03, seed=123, bins=bins)
    mec2 = 0.511
    e_edge = e0 - e_sc_180
    plt.figure(figsize=(9.5, 6.0))
    plt.step(centers, hist, where="mid")
    plt.axvline(e0, color="r", linestyle="--", alpha=0.6)
    plt.axvline(e_edge, color="g", linestyle="--", alpha=0.6)
    plt.axvline(e_sc_180, color="C1", linestyle="--", alpha=0.6)
    plt.xlabel("E / MeV")
    plt.ylabel("entries")
    plt.title("Simulated gamma spectrum (E0=2 MeV)")
    plt.tight_layout()
    plt.savefig("simulate-2MeV.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
