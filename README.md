# XOR Neural Network for Commodore 64

A simple neural network implementation in pure C that runs on the Commodore 64 and learns the XOR function using backpropagation.

## XOR


```
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
```

XOR is significant in neural network history because it cannot be solved by a single-layer perceptron (a linear classifier). It requires a hidden layer, making it a classic test for multi-layer networks.

## Network Architecture

This implementation uses a simple feedforward neural network:

- **Input layer**: 2 neurons (for the two binary inputs)
- **Hidden layer**: 4 neurons (with tanh-like activation)
- **Output layer**: 1 neuron (produces a value between 0 and 1)

```
Input (2) → Hidden (4) → Output (1)
```

## How It Works

### 1. Fixed-Point Arithmetic

The C64 has no floating-point unit, so all calculations use integer math with a scale factor of 100. For example:
- 0.50 is represented as 50
- 1.00 is represented as 100
- Weights like 0.75 are stored as 75

### 2. Forward Propagation

For each input pattern, the network:

1. Calculates hidden layer activations:
   ```
   h[j] = activate(b1[j] + x[0]*w1[0][j] + x[1]*w1[1][j])
   ```

2. Calculates output:
   ```
   out = activate(b2 + sum(h[i] * w2[i]))
   ```

The `activate()` function is a simplified tanh approximation that maps values to 0-100 range.

### 3. Backpropagation

The network learns by adjusting weights to minimize error:

1. **Calculate output error**: `error = target - output`

2. **Compute output gradient**: `d_out = error * deriv(output)`

3. **Compute hidden gradients**: Each hidden neuron's gradient is based on how much it contributed to the output error

4. **Update weights**: Add gradients (scaled by learning rate) to weights:
   ```
   w2[i] += learning_rate * d_out * h[i]
   w1[i][j] += learning_rate * d_h[j] * x[i]
   ```

### 4. Training Process

The network trains for 8000 epochs. In each epoch:
- All 4 XOR patterns are presented
- Forward propagation computes output
- Backpropagation adjusts weights
- Process repeats until weights converge

## Key Design Decisions

### Activation Function

Instead of a true sigmoid or tanh, a piecewise linear approximation is used:

```c
int activate(int x) {
    if (x > 200) return SCALE;
    if (x < -200) return 0;
    return 50 + (x / 4);  // Linear region
}
```

This provides:
- Fast computation (no exponentials)
- Non-zero gradients in the learning region
- Sufficient non-linearity to learn XOR

### Derivative Function

The derivative returns a constant value in the learning region:

```c
int deriv(int y) {
    if (y < 10 || y > 90) return 5;
    return 25;
}
```

This ensures gradients don't vanish to zero after integer division.

### Learning Rate

The effective learning rate is approximately 1.0, which is aggressive but necessary because:
- Integer division truncates values
- Small gradients would vanish to zero
- The C64's limited precision requires larger updates

## Compilation

Using the cc65 compiler toolchain:

```bash
make
```

Creates a prg file and a d64 disk images that can be loaded on your C64 or emulator (like VICE).

## Expected Results

After training, the network should output:

```
0 XOR 0 = 0.08 - 0.12  (close to 0)
0 XOR 1 = 0.88 - 0.95  (close to 1)
1 XOR 0 = 0.88 - 0.95  (close to 1)
1 XOR 1 = 0.08 - 0.12  (close to 0)
```

The exact values vary due to random weight initialization.

## Limitations

- **Integer precision**: Results won't be as accurate as floating-point implementations
- **Speed**: Training takes several seconds on a 1MHz CPU
- **Memory**: Uses about 200 bytes for weights and activations
- **Simplified math**: Activation functions are approximations

## Why This Is Interesting

This demonstrates that:
- Neural networks can run on extremely limited hardware
- Fixed-point arithmetic can train networks successfully
- The C64 (from 1982) can perform machine learning
- Careful numerical design is crucial for resource-constrained systems

