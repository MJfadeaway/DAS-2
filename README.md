# DAS-2: Generalizing DAS for surrogate modeling of parametric differential equations
 Official implementation for the paper [Deep adaptive sampling for surrogate modeling without labeled data](https://arxiv.org/abs/2402.11283)

We propose a deep adaptive sampling approach for surrogate modeling of parametric differential equations without labeled data, i.e., DAS for surrogates ($\text{DAS}^2$).
We demonstrate the efficiency of the proposed method with a series of numerical experiments, including the operator learning problem, the parametric optimal control problem,
and the lid-driven 2D cavity flow problem with a continuous range of Reynolds numbers from 100 to 3200 (will be released soon). 


# Requirements

PyTorch, 
Numpy, 
Scipy



# Motivation
Surrogate modeling is of great practical significance for parametric differential equation systems. In contrast to classical numerical methods, using physics-informed deep learning-based methods to construct simulators for such systems is a promising direction due to its potential to handle high dimensionality, which requires minimizing a loss over a training set of random samples. However, the random samples introduce statistical errors, which may become the dominant errors for the approximation of low-regularity and high-dimensional problems.



# Train
Operator learning
```bash
python das_oplearning.py
```

Surrogate modeling for parametric optimal control
```bash
python das_train.py
```



# Citation
If you find this repo useful for your research, please consider to cite our paper (arXiv, 2024)
```
@article{wang2024deep,
  title={Deep adaptive sampling for surrogate modeling without labeled data},
  author={Wang, Xili and Tang, Kejun and Zhai, Jiayu and Wan, Xiaoliang and Yang, Chao},
  journal={arXiv preprint arXiv:2402.11283},
  year={2024}
}
```
