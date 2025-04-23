Source code for the paper "Identifying Large-Scale Linear Parameter Varying Systems with Dynamic Mode Decomposition Methods", submitted in the Journal of Computational Physics. 
Pre-print is currently available on arXiv at https://arxiv.org/abs/2502.02336.

The main file with the DMD-LPV implementation is lovated on "lpvs_ident.py"

The model used as the main case study is implemented at "lineardiffusion_eq.py". Run this file before starting the global experiments.

The experiment for section 6.4 ia implwmented in lineardiffusion_ident_fullinfo.py for the exact model, lineardiffusion_ident_overestimated.py, for teh 4th degree polynomial model, and lineardiffusion_ident_underestimated.py
for the second degree model.

For section 6.5, the files lineardiffusion_ident_POD.py, lineardiffusion_ident_POD_under.py lineardiffusion_ident_POD_over.py plot the experiments. Figure 2 is obtained by under_and_over_POD_plot.py

Section 6.6 requires the user to run lineardiffusion_localident.py first, and it simulates the system and provides training data arranged locally.

Then, lineardiffusion_localident_training.py is responsible for the local training. The three cases are obtained through switching the commented polynomials in k_fun.

To change the traning method, local_train() refers to Algorithm 2, while local_trian_alt() refers to Algorithm 3. All data on Table 2 and 3 were obtained from this file.

Section 6.7 has its experiments on test data generated from generate_results_showcase.py. The experiments themselves are run on results_showcase.py and plotted on simulated_results.py


