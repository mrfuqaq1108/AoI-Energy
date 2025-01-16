# Related Paper
- Baojie Fu, Dapeng Wu and Ruyan Wang, Analysis on Resource Allocation in Image
Surveillance System

***Any use of the codes should explicitly cite the aforementioned paper.***

# Dataset
- Animals-10, which is available online at  https://www.kaggle.com/datasets/alessiocorrado99/animals10/data/

# Documentation

## Data Fitting
refer to https://github.com/mrfuqaq1108/TAoII-Image-Transmission

- **figure_nn_loss.py**: Print figure (Loss&Accuracy versus Epoch Number)**(Figure 3)**
- **diff_SNR.py**: Calculate the image recognition accuracy under different probabilities of successful decoding.

## code for image surveillance
- **main_final.py**: Code for Algorithm 1, Intelligent Joint Computing Scheme
- **solve_function.py**: Some function defined for main_final.py.

## simulation results
- **compare_Task.py**: FIGURE 6, task performance comparision under different schemes, q1=0.8, q2=0.2.
- **compare_PAoI.py**: FIGURE 7, PAoI performance comparision under different schemes, q1=0.8, q2=0.2.
- **compare_Energy.py**: FIGURE 8, energy consumption performance comparision under different schemes, q1=0.8, q2=0.2.
- **compare_different_M.py**: FIGURE 9, average task value with different number of cameras under different schemes, q1 = 0.8, q2 = 0.2.
- **diff_f.py**: FIGURE 10, average task value with different average CPU frequency of cameras under different schemes, q1 = 0.8, q2 = 0.2.
- **diff_C_delta.py**: FIGURE 11, Average task value with different required number of CPU cycles for image under different schemes, q1 = 0.8, q2 = 0.2.
- **diff_B.py**: FIGURE 12, Average task value with different bandwidth under different schemes, q1 = 0.8, q2 = 0.2.
- **diff_sigma.py**: FIGURE 13, Average task value with different noise power under different schemes, q1 = 0.8, q2 = 0.2.
- **diff_acc.py**: FIGURE 14, Average task value with different image recognition accuracy under different schemes, q1 = 0.8, q2 = 0.2.