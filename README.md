# Koopman Forecasting package
The problem statement can be found [here](reports/first_report.pdf)
#### Baseline solutions
For short-term forecasting, the following [algorithm](https://github.com/erichson/koopmanAE) demonstrated the state-of-the-art. The work requires the consistency of the system, that is, the ability to make predictions in both directions. This minimizes the following loss
$$ε = λ_{id}ε_{id} + λ_{fwd}ε_{fwd} + λ_{bwd}ε_{bwd} + λ_{con}ε_{con},$$
where $$ε_{id}$$ is a reconstruction loss. $$ε_{fwd}, ε_{bwd}$$ – k steps forward (backward) prediction error. $$ε_{con}$$ – consistency loss.
For long-term forecasting, there was proposed [Spectral Methods usage with Koopman theory](https://github.com/helange23/from_fourier_to_koopman), and a comparison with Fourier transform is made.

#### Roles for the participants (preliminary)
Nikita Balabin (50%) – Refactoring the two methods with PyTorch Lightning as the main framework for the library. Introduction of Ray to optimize all hyperparameters. + Main implementations.
Oleg Maslov (50%) – Introduction the unit testing and provide a test coverage of 70% of the codebase. Creation of the necessary documentation for the API using readthedocs. Providing notebooks with examples on how to run each method. + Main implementations.
