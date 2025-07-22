from grgapp.assoc import (
    read_covariates_matrix, read_pheno, linear_assoc_covar, linear_assoc_no_covar,
)
import pygrgl
import numpy as np
import time
import pandas as pd
sample_sizes = [10, 100, 1000, 10000, 100000, 500000]

grg_paths = {
    10:   "/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-10-100000000.igd.final.grg",
    100:  "/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-100-100000000.igd.final.grg",
    1000: "/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-1000-100000000.igd.final.grg",
    10000:"/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-10000-100000000.igd.final.grg",
    100000:"/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-100000-100000000.igd.final.grg",
    500000:"/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-500000-100000000.igd.final.grg"
}


covariate_counts = [1, 5, 10, 20]

results = []

###PLAN
# 5) Loop over each sample‐size and each covariate‐count.
#    a) Load the GRG with pygrgl.load_immutable_grg(...)  (or your preferred loader).
#    b) Generate a random phenotype Y of length = sample_size.
#    c) Generate a random covariate matrix C of shape (sample_size × n_covariates).
#    d) Time compute_GWAS_COV(grg, Y, C).
#    e) Record the elapsed time.


for n in sample_sizes:
    grg_file = grg_paths[n]

    grg = pygrgl.load_immutable_grg(grg_file)

    for p in covariate_counts:
        Y = np.random.randn(n)

        intercept = np.ones((n,1))
        random_covs = np.random.randn(n, p)
        C = np.hstack([intercept, random_covs])


        # 3) Time how long compute_GWAS_COV takes
        t0 = time.perf_counter()
        betas = linear_assoc_covar(grg, Y, C) 
        t1 = time.perf_counter()

        elapsed = t1 - t0
        results.append({
            "sample_size": n,
            "n_covariates": p,
            "runtime_seconds": elapsed
        })

# 6) Convert to pandas.DataFrame
df = pd.DataFrame(results)
df.to_csv("gwas_timing_results2.csv", index=False)

print(df)