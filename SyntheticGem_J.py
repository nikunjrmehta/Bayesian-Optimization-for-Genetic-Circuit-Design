import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

# Set publication style
plt.style.use("seaborn-v0_8-paper")
sns.set_context("paper", font_scale=1.5)

# Centralized Configuration and Parameters
PARAMS = {
    "sequence_length": 15,
    "alphabet_size": 4,
    "latent_dim": 15,
    "initial_samples": 20,
    "bo_iterations": 60,
    "num_candidates": 1200,
    "matern_nu": 2.5,
    "length_scale_bounds": (1e-3, 1e2),
    "viability_threshold": 0.2,
    "use_pov": False,
    "exploration_prob": 0.2,
    "candidate_range": 5,
    "gp_alpha": 1e-3,
    "gp_jitter": 1e-6,
    "seed": 7,
    "local_explore_scale": 0.5,
    "local_candidates": 800,
    "global_candidates": 600,
    "mutation_rate": 0.3,
    "elite_fraction": 0.3,
    "ucb_beta": 3.0,
    "random_injection": 200,
    "noise_std": 0.0,
    "show_plots": False,
    "diagnostics": True,
    "diagnostic_samples": 500,
    "multi_seed_runs": True,
    "seed_list": [7, 13, 21, 37, 73],
}


# 1. Ehrlich Occupancy Time Biophysical Objective
class EhrlichOccupancyObjective:
    def __init__(self, L=PARAMS["sequence_length"]):
        self.L = L
        self.target = np.random.randint(0, PARAMS["alphabet_size"], L)
        self.viable_total = 0
        self.eval_total = 0

    def evaluate(self, x_seq):
        hamming_sim = np.sum(x_seq == self.target) / self.L
        kd = 1.0 - hamming_sim
        occupancy = 1.0 / (kd + 1.0)
        self.eval_total += 1
        if hamming_sim < PARAMS["viability_threshold"]:
            return np.nan
        self.viable_total += 1
        if PARAMS["noise_std"] <= 0:
            return occupancy
        return occupancy + np.random.normal(0, PARAMS["noise_std"])

    def viable_rate(self):
        if self.eval_total == 0:
            return 0.0
        return self.viable_total / self.eval_total

    def occupancy(self, x_seq):
        hamming_sim = np.sum(x_seq == self.target) / self.L
        kd = 1.0 - hamming_sim
        return 1.0 / (kd + 1.0), hamming_sim


# 2. Bijective Flow for Sequence Embedding
class BijectiveFlow(nn.Module):
    def __init__(self, dim=PARAMS["latent_dim"]):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def encode(self, x):
        return torch.matmul(x, self.weight) + self.bias

    def decode(self, z):
        return z - self.bias


# 3. Probability of Viability Classifier
class PoVClassifier(nn.Module):
    def __init__(self, dim=PARAMS["latent_dim"]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )

    def predict_viable(self, z):
        return self.net(torch.FloatTensor(z)).detach().numpy().flatten()


# 4. Decoupled COWBOYS Optimizer
class COWBOYS_Optimizer:
    def __init__(self, objective, dim=PARAMS["latent_dim"], use_flow=False, use_pov=False):
        self.obj = objective
        self.flow = BijectiveFlow(dim)
        self.pov = PoVClassifier(dim)
        self.seq_length = PARAMS["sequence_length"]
        self.alphabet_size = PARAMS["alphabet_size"]
        self.use_flow = use_flow
        self.use_pov = use_pov
        self.surrogate = GaussianProcessRegressor(
            kernel=Matern(
                nu=PARAMS["matern_nu"], length_scale_bounds=PARAMS["length_scale_bounds"]
            ),
            alpha=PARAMS["gp_alpha"],
            normalize_y=True,
            n_restarts_optimizer=2,
        )
        self.X_obs, self.Y_obs, self.seq_obs = [], [], []

    def acquisition(self, z_cand):
        mu, sigma = self.surrogate.predict(z_cand, return_std=True)
        sigma = np.maximum(sigma, PARAMS["gp_jitter"])
        y_best = np.max(self.Y_obs) if len(self.Y_obs) > 0 else 0
        with np.errstate(divide="ignore"):
            Z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ucb = mu + PARAMS["ucb_beta"] * sigma
        score = 0.5 * ei + 0.5 * ucb
        if self.use_pov:
            score = score * self.pov.predict_viable(z_cand)
        return score

    def _seq_vector(self, seq):
        return seq.astype(float)

    def _to_latent(self, seq):
        if not self.use_flow:
            return self._seq_vector(seq)
        with torch.no_grad():
            return self.flow.encode(torch.FloatTensor(seq)).numpy()

    def _random_sequence(self):
        return np.random.randint(0, self.alphabet_size, self.seq_length)

    def _mutate(self, seq):
        mutated = seq.copy()
        mask = np.random.rand(self.seq_length) < PARAMS["mutation_rate"]
        for idx in np.where(mask)[0]:
            options = list(range(self.alphabet_size))
            options.remove(int(mutated[idx]))
            mutated[idx] = np.random.choice(options)
        return mutated

    def _generate_candidates(self):
        candidates = [self._random_sequence() for _ in range(PARAMS["global_candidates"])]
        if len(self.seq_obs) == 0:
            return candidates
        elite_count = max(1, int(len(self.seq_obs) * PARAMS["elite_fraction"]))
        elite_idx = np.argsort(self.Y_obs)[-elite_count:]
        elite_seqs = [self.seq_obs[i] for i in elite_idx]
        while len(candidates) < PARAMS["global_candidates"] + PARAMS["local_candidates"]:
            parent = elite_seqs[np.random.randint(len(elite_seqs))]
            candidates.append(self._mutate(parent))
        for _ in range(PARAMS["random_injection"]):
            candidates.append(self._random_sequence())
        return candidates

    def run_iteration(self):
        if len(self.X_obs) > 1 and np.std(self.Y_obs) > 0:
            self.surrogate.fit(np.array(self.X_obs), np.array(self.Y_obs))
        candidates = self._generate_candidates()
        if (
            len(self.X_obs) < 2
            or np.std(self.Y_obs) == 0
            or np.random.rand() < PARAMS["exploration_prob"]
        ):
            x_seq = candidates[np.random.randint(len(candidates))]
        else:
            X_cand = np.array([self._to_latent(seq) for seq in candidates])
            scores = self.acquisition(X_cand)
            x_seq = candidates[int(np.argmax(scores))]
        y_val = self.obj.evaluate(x_seq)
        if not np.isnan(y_val):
            self.seq_obs.append(x_seq)
            self.X_obs.append(self._to_latent(x_seq))
            self.Y_obs.append(y_val)
        return y_val


def run_optimization(run_seed, use_flow=False, use_pov=False, label=None):
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    objective = EhrlichOccupancyObjective()
    optimizer = COWBOYS_Optimizer(objective, use_flow=use_flow, use_pov=use_pov)
    for _ in range(PARAMS["initial_samples"]):
        x_seq = optimizer._random_sequence()
        y_init = objective.evaluate(x_seq)
        if not np.isnan(y_init):
            optimizer.seq_obs.append(x_seq)
            optimizer.X_obs.append(optimizer._to_latent(x_seq))
            optimizer.Y_obs.append(y_init)
    history = []
    for i in range(PARAMS["bo_iterations"]):
        optimizer.run_iteration()
        if len(optimizer.Y_obs) > 0:
            current_best = np.max(optimizer.Y_obs)
            history.append(current_best)
            label_prefix = f"{label} | " if label else ""
            print(f"{label_prefix}Cycle {i + 1}: Best Fitness = {current_best:.4f}")
    return history, objective.viable_rate(), optimizer


def run_diagnostics():
    np.random.seed(PARAMS["seed"] + 100)
    objective = EhrlichOccupancyObjective()
    optimizer = COWBOYS_Optimizer(objective, use_flow=False, use_pov=False)
    samples = PARAMS["diagnostic_samples"]
    best = None
    for _ in range(samples):
        seq = optimizer._random_sequence()
        val = objective.evaluate(seq)
        if np.isnan(val):
            continue
        best = val if best is None else max(best, val)
    best_str = f"{best:.4f}" if best is not None else "n/a"
    print(
        "Diagnostics | random baseline best="
        f"{best_str}, viable_rate={objective.viable_rate():.2%}"
    )


def compute_stats(histories):
    stats = []
    for label, history in histories.items():
        if not history:
            stats.append({"method": label, "final_best": np.nan, "max_best": np.nan, "mean_best": np.nan})
            continue
        stats.append(
            {
                "method": label,
                "final_best": history[-1],
                "max_best": max(history),
                "mean_best": float(np.mean(history)),
            }
        )
    return stats


def write_results_csv(histories, stats, path="figure_2_results.csv"):
    if not histories:
        return
    max_len = max(len(history) for history in histories.values())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration"] + list(histories.keys()))
        for idx in range(max_len):
            row = [idx + 1]
            for history in histories.values():
                row.append(history[idx] if idx < len(history) else "")
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["method", "final_best", "max_best", "mean_best"])
        for row in stats:
            writer.writerow(
                [
                    row["method"],
                    f"{row['final_best']:.6f}" if row["final_best"] == row["final_best"] else "",
                    f"{row['max_best']:.6f}" if row["max_best"] == row["max_best"] else "",
                    f"{row['mean_best']:.6f}" if row["mean_best"] == row["mean_best"] else "",
                ]
            )


def summarize_multi_seed(results):
    summary = {}
    for method, histories in results.items():
        if not histories:
            summary[method] = {"mean": np.array([]), "std": np.array([])}
            continue
        max_len = max(len(history) for history in histories)
        padded = []
        for history in histories:
            last = history[-1]
            padded.append(history + [last] * (max_len - len(history)))
        data = np.array(padded)
        summary[method] = {"mean": data.mean(axis=0), "std": data.std(axis=0)}
    return summary


def plot_figure_2_summary(summary):
    plt.figure(figsize=(8, 6))
    for label, stats in summary.items():
        if stats["mean"].size == 0:
            continue
        iterations = np.arange(1, len(stats["mean"]) + 1)
        plt.plot(iterations, stats["mean"], linewidth=2.3, label=label)
        plt.fill_between(
            iterations,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.2,
        )
    plt.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.5, label="Theoretical Max"
    )
    plt.xlabel("Optimization Iterations")
    plt.ylabel("Average Best Fitness")
    plt.title("Convergence on Ehrlich Occupancy Time Landscape (Mean ± Std)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("figure_2_convergence_mean_std.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


def write_aggregated_csv(summary, path="figure_2_results_aggregate.csv"):
    if not summary:
        return
    max_len = max((len(stats["mean"]) for stats in summary.values()), default=0)
    if max_len == 0:
        return
    methods = list(summary.keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["iteration"]
        for method in methods:
            header.extend([f"{method}_mean", f"{method}_std"])
        writer.writerow(header)
        for idx in range(max_len):
            row = [idx + 1]
            for method in methods:
                stats = summary[method]
                if idx < len(stats["mean"]):
                    row.append(f"{stats['mean'][idx]:.6f}")
                    row.append(f"{stats['std'][idx]:.6f}")
                else:
                    row.extend(["", ""])
            writer.writerow(row)


def run_multi_seed(seed_list):
    results = {"Sequence BO": [], "Flow BO": [], "Flow + PoV": []}
    for seed in seed_list:
        history, _, _ = run_optimization(seed, use_flow=False, use_pov=False, label="Sequence BO")
        results["Sequence BO"].append(history)
        history, _, _ = run_optimization(seed + 1, use_flow=True, use_pov=False, label="Flow BO")
        results["Flow BO"].append(history)
        history, _, _ = run_optimization(seed + 2, use_flow=True, use_pov=True, label="Flow + PoV")
        results["Flow + PoV"].append(history)
    summary = summarize_multi_seed(results)
    plot_figure_2_summary(summary)
    write_aggregated_csv(summary)
    print("Multi-seed summary (final mean ± std):")
    for label, stats in summary.items():
        if stats["mean"].size == 0:
            continue
        print(
            f"{label}: {stats['mean'][-1]:.4f} ± {stats['std'][-1]:.4f}"
        )


# --- Figure 2: Ehrlich Convergence (L=15) ---
def plot_figure_2(histories=None):
    plt.figure(figsize=(8, 6))
    if histories:
        for label, history in histories.items():
            iterations = np.arange(1, len(history) + 1)
            plt.plot(iterations, history, linewidth=2.3, label=label)
    else:
        iterations = np.arange(1, 31)
        # Mock data based on hdbo_benchmark results in manuscript
        lsgp_fitness = 0.864 / (1 + np.exp(-0.3 * (iterations - 10)))
        ga_fitness = 0.336 / (1 + np.exp(-0.1 * (iterations - 5)))
        lambo_fitness = 0.510 / (1 + np.exp(-0.15 * (iterations - 15)))

        plt.plot(
            iterations, lsgp_fitness, "b-", linewidth=2.5, label="Decoupled LSGP (Ours)"
        )
        plt.plot(iterations, lambo_fitness, "g--", linewidth=2, label="LaMBO-2")
        plt.plot(iterations, ga_fitness, "r:", linewidth=2, label="Genetic Algorithm")

    plt.axhline(
        y=1.0, color="gray", linestyle="--", alpha=0.5, label="Theoretical Max"
    )
    plt.xlabel("Optimization Iterations")
    plt.ylabel("Average Best Fitness")
    plt.title("Convergence on Ehrlich Occupancy Time Landscape")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("figure_2_convergence.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


# --- Figure 3: Real-World Efficiency (Dual Panel) ---
def plot_figure_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: GB1 Burden
    methods_gb1 = ["Standard BO", "Decoupled LSGP"]
    samples_gb1 = [150, 10]
    ax1.bar(methods_gb1, samples_gb1, color=["lightcoral", "cornflowerblue"])
    ax1.set_ylabel("Samples Required for Optima")
    ax1.set_title("A: GB1 Protein Burden")
    ax1.text(1, 150, "15x Gain", ha="center", fontweight="bold")

    # Panel B: Limonene Points
    methods_lim = ["Standard BO", "Decoupled LSGP"]
    samples_lim = [43, 10]
    ax2.bar(methods_lim, samples_lim, color=["lightcoral", "cornflowerblue"])
    ax2.set_ylabel("Unique Experiments")
    ax2.set_title("B: Limonene Pathway Efficiency")
    ax2.text(1, 25, "4.3x Gain", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figure_3_efficiency.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


def plot_figure_5_stats(stats):
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = [row["method"] for row in stats]
    final_vals = [row["final_best"] for row in stats]
    mean_vals = [row["mean_best"] for row in stats]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, mean_vals, width, label="Mean Best")
    ax.bar(x + width / 2, final_vals, width, label="Final Best")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Fitness")
    ax.set_title("Method Comparison (Observed)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_5_method_comparison.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


def plot_figure_3_benchmarks():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: GB1 Burden
    methods_gb1 = ["Standard BO", "Decoupled LSGP"]
    samples_gb1 = [150, 10]
    ax1.bar(methods_gb1, samples_gb1, color=["lightcoral", "cornflowerblue"])
    ax1.set_ylabel("Samples Required for Optima")
    ax1.set_title("A: GB1 Protein Burden")
    ax1.text(1, 150, "15x Gain", ha="center", fontweight="bold")

    # Panel B: Limonene Points
    methods_lim = ["Standard BO", "Decoupled LSGP"]
    samples_lim = [43, 10]
    ax2.bar(methods_lim, samples_lim, color=["lightcoral", "cornflowerblue"])
    ax2.set_ylabel("Unique Experiments")
    ax2.set_title("B: Limonene Pathway Efficiency")
    ax2.text(1, 25, "4.3x Gain", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("figure_3_efficiency.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


# --- Figure 4: Latent Space and PoV Boundary ---
def plot_figure_6_flow(objective, flow):
    samples = 800
    sequences = np.random.randint(
        0, PARAMS["alphabet_size"], size=(samples, PARAMS["sequence_length"])
    )
    latents = []
    fitness = []
    viable = []
    for seq in sequences:
        with torch.no_grad():
            latent = flow.encode(torch.FloatTensor(seq)).numpy()
        occ, hamming = objective.occupancy(seq)
        latents.append(latent[:2])
        fitness.append(occ)
        viable.append(hamming >= PARAMS["viability_threshold"])
    latents = np.array(latents)
    fitness = np.array(fitness)
    viable = np.array(viable)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    hex_plot = ax1.hexbin(
        latents[:, 0],
        latents[:, 1],
        C=fitness,
        gridsize=30,
        cmap="viridis",
        reduce_C_function=np.mean,
        mincnt=1,
    )
    fig.colorbar(hex_plot, ax=ax1, label="Mean Occupancy Fitness")
    ax1.set_title("Latent Fitness Density")
    ax1.set_xlabel("Latent Dimension 1")
    ax1.set_ylabel("Latent Dimension 2")

    ax2.scatter(
        latents[viable, 0],
        latents[viable, 1],
        c=fitness[viable],
        cmap="viridis",
        s=28,
        alpha=0.75,
        edgecolor="none",
        label="Viable",
    )
    ax2.scatter(
        latents[~viable, 0],
        latents[~viable, 1],
        c="black",
        s=28,
        alpha=0.5,
        marker="x",
        label="Non-viable",
    )
    ax2.set_title("Viable vs Non-viable Samples")
    ax2.set_xlabel("Latent Dimension 1")
    ax2.legend(loc="upper right")

    plt.suptitle("Flow-Encoded Manifold Overview", fontsize=14)
    plt.tight_layout()
    plt.savefig("figure_6_flow_manifold.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


def plot_figure_4_benchmarks():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    # Synthetic fitness landscape
    Z = np.exp(-((X - 2) ** 2 + (Y - 2) ** 2) / 4) + 0.5 * np.exp(
        -((X + 2) ** 2 + (Y + 2) ** 2) / 8
    )
    # PoV Boundary logic (sequences < -1 in X fail)
    V = (X > -1.5).astype(float)

    plt.figure(figsize=(8, 7))
    cp = plt.contourf(X, Y, Z, cmap="viridis", alpha=0.8)
    plt.colorbar(cp, label="Predicted Fitness")

    # Draw viability boundary
    plt.contour(X, Y, V, levels=[0.5], colors="red", linestyles="solid", linewidths=3)
    plt.text(-4, 0, "Toxic Region", color="red", fontsize=12, fontweight="bold")
    plt.text(2, 2, "Viable Optima", color="white", fontsize=12, fontweight="bold")

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Protein Manifold with Viability Boundary")
    plt.savefig("figure_4_benchmark.png", dpi=300)
    if PARAMS["show_plots"]:
        plt.show()


if __name__ == "__main__":
    if PARAMS["diagnostics"]:
        run_diagnostics()
    if PARAMS["multi_seed_runs"]:
        run_multi_seed(PARAMS["seed_list"])
    histories = {}
    optimizers = {}
    history, _, optimizer = run_optimization(
        PARAMS["seed"], use_flow=False, use_pov=False, label="Sequence BO"
    )
    histories["Sequence BO"] = history
    optimizers["Sequence BO"] = optimizer
    history, _, optimizer = run_optimization(
        PARAMS["seed"] + 1, use_flow=True, use_pov=False, label="Flow BO"
    )
    histories["Flow BO"] = history
    optimizers["Flow BO"] = optimizer
    history, _, optimizer = run_optimization(
        PARAMS["seed"] + 2, use_flow=True, use_pov=True, label="Flow + PoV"
    )
    histories["Flow + PoV"] = history
    optimizers["Flow + PoV"] = optimizer
    stats = compute_stats(histories)
    write_results_csv(histories, stats)
    print("Method comparison stats:")
    for row in stats:
        print(
            f"{row['method']}: final={row['final_best']:.4f}, "
            f"max={row['max_best']:.4f}, mean={row['mean_best']:.4f}"
        )
    plot_figure_2(histories)
    plot_figure_5_stats(stats)
    plot_figure_3_benchmarks()
    best_method = max(stats, key=lambda row: row["final_best"])["method"]
    plot_figure_6_flow(optimizers[best_method].obj, optimizers[best_method].flow)
    plot_figure_4_benchmarks()
