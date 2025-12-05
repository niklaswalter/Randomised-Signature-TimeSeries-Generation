timeseries: 
  n_lags: 10
  data_dim: 1
  p: 6
  q: 10
bm:
  samples: 100000
  drift: 0.1
  std: 0.2
gbm:
  samples: 50000
  drift: 0.0
  std: 1.0
  initial_value_gbm: 1.0
ar:
  samples: 50000
  phi: -0.1
  std: 1.0
hyperparameters:
  learning_rate: 1e-4
  gradient_steps: 5
  batch_size: 10000
  mc_num: 1000
rsigw1: 
  reservoir_dim_metric: 80
sigw1:
  truncation_depth: 4
  normalise: True
neural_sde:
  input_dim: 32
  hidden_dim: 32
  brownian_dim: 1
  reservoir_dim_gen: 80
  activation: "Sigmoid"
lstm:
  inut_dim: 5
  hidden_dim: 64
  num_layers: 2
data: 
  id: "BM"
generator:
  id: "NeuralSDE"
discriminator:
  id: "RSigW1"
others: 
  trainable_var: True
  same_matrices: False
  time_homogeneous_readout: False
