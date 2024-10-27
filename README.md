# Neural Network Initialization Experimentation

This project demonstrates the impact of various weight initialization strategies on a neural network's performance using a binary classification task on a custom dataset.

## Project Structure

The project includes the following components:

1. **Zero Initialization**: 
    - Network architecture: 2-layer feedforward neural network
    - Activation functions: ReLU and Sigmoid
    - Experiment with all weights initialized to zero.
    - Observes the effect of zero initialization on convergence and learning.

2. **Non-Zero Constant Initialization**:
    - Initializes weights with a constant value (0.5) for all layers.
    - Uses the Sigmoid activation function.
    - Observes model behavior when all weights start with the same non-zero constant.

3. **Random Initialization**:
    - **Small Weights**: Randomly initializes weights with small values.
    - **Large Weights**: Randomly initializes weights with larger values.

4. **Standard Initialization Techniques**:
    - **Xavier/Glorot Initialization** (used with Tanh activation): Sets weights to values drawn from a distribution that scales based on input/output layer sizes.
    - **He Initialization** (used with ReLU activation): Uses weight scaling specifically for ReLU networks to prevent gradient vanishing/exploding issues.

## Repository Files

- `ushape.csv`: Sample dataset used for binary classification.
- `model_initialization_experiments.ipynb`: Jupyter notebook containing code for all initialization experiments.
- `README.md`: Documentation of the project and initialization strategies.

## Dependencies

- Python 3.x
- Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, mlxtend

You can install all dependencies by running:

```bash
pip install tensorflow numpy pandas matplotlib mlxtend
```

## Usage

To run the project:

1. Load the dataset by placing it in the `/input/` folder.
2. Run the cells to initialize, compile, and fit each model on the dataset.
3. Observe and compare decision boundary plots for different initialization strategies.

```python
import pandas as pd
df = pd.read_csv('/kaggle/input/ushape/ushape.csv')
```

## Results

Each model's performance can be visualized by plotting decision boundaries. The plots are generated using `mlxtend.plotting.plot_decision_regions` after training to illustrate how initialization affects classification boundaries.

### Key Observations

1. **Zero Initialization**: Models fail to learn, resulting in poor accuracy.
2. **Constant Initialization**: Model's convergence rate is hindered due to symmetry in weights.
3. **Random Initialization**: Shows better results, especially with small random weights.
4. **Xavier/Glorot and He Initialization**: Improved accuracy, faster convergence.

## Future Work

Experiment with additional initialization techniques and test them on larger datasets or different architectures to observe generalization capabilities.

## License

MIT License
