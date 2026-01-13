## Understanding How Neural Networks Learn

This document explains the complete training cycle of machine learning models.

> **Important Note:** These steps represent a common framework for understanding model training, but they are **not fixed or universal**. Different models, architectures, and training paradigms may modify, skip, or add steps based on their specific requirements.

---

## Steps of Model Training

#### Step 1: Weight Initialization

We start with training the model because **the model has random weights and knows absolutely nothing about the task**. We must expose it to data so it can learn patterns.

Think of this as a student walking into their first day of class with no prior knowledge.



#### Step 2: Forward Pass (*Prediction*)

The model then makes predictions because **learning only happens when we compare what it thinks with what is actually correct**.

This is the model's attempt at solving the problem based on its current understanding.



#### Step 3: Loss per Sample

We calculate the loss for each data point because **each example carries its own error**. Without measuring individual mistakes, the model has no feedback.

Every single prediction gets evaluated independently to understand where the model went wrong.



#### Step 4: Cost Function (*Batch Loss*)

We take the **average of all losses in a batch** to get a single scalar called the **cost**. This is what the optimizer will minimize.

##### Example
If a batch has 32 samples, the cost = mean(loss₁, loss₂, ..., loss₃₂)

> **Why?** The optimizer cannot work with thousands of separate errors; it needs one scalar value to minimize.



#### Step 5: Backpropagation (*Gradient Computation*)

Now we perform backpropagation because **we need to know which weights caused how much error**.

Backpropagation uses the **chain rule** to compute derivatives of the cost with respect to every weight. This tells us the **direction** in which each weight should move to reduce the error.

*This is the mathematical heart of neural network learning.*



#### Step 6: Optimization Step (*Parameter Update*)

After that, the optimizer updates the weights because **derivatives alone do nothing**.

The optimizer takes the gradients and actually moves the weights a small step in the direction that lowers the cost.

- **Gradients** tell us the direction
- **Learning rate** (and optimizer strategy like momentum or Adam) determines how big the step is



#### Step 7: Training Loop / Iterative Learning

We repeat this entire cycle many times because **the model does not find the optimal solution in one step**.

Each update only improves the model slightly, and only after **thousands of these small corrections** does the model approach the lowest point of the cost surface.

---

## Batches and Gradient Descent

### How Often Do Weights Update?

Usually each weight update happens **after every batch, not after every epoch**. This is called **mini-batch gradient descent**, which is the standard in deep learning.

### Three Different Strategies

1. **Mini-batch GD** → Updates after each batch (most common)
2. **Stochastic GD (SGD)** → Updates after each single sample (batch size = 1)
3. **Batch GD** → Updates after the entire dataset (rare, too slow for large data)

---

## Understanding Epochs vs Batches

##### Epoch Definition >> An **epoch** means the model has seen the entire dataset once.

##### Why Split into Batches?
The dataset is split into batches because updating after all data would be too slow and too memory-heavy.

##### The Process for Every Batch
The model predicts → loss is computed → cost is calculated → backprop computes gradients → the optimizer updates the weights

---

## Practical Example

If you have **1,000 samples** and **batch size = 100**, then:

- One epoch has **10 batches**
- The model updates its weights **10 times per epoch**
- If you train for **20 epochs**, total updates = **200**

---

## Machine Learning vs Deep Learning

These steps happen in both Machine Learning and Deep Learning, **but** backpropagation is specific to neural networks which are the heart of Deep Learning.

### Comparison Table

| Step | Exists in ML | Exists in DL | Notes |
|------|-------------|--------------|-------|
| Weight initialization | Sometimes | Always | Random init is core in NN |
| Forward pass | Yes | Yes | All supervised learners predict |
| Per-sample loss | Yes | Yes | Needed to measure error |
| Cost function | Yes | Yes | One scalar objective |
| **Backpropagation** | No | Yes | Neural networks only (though some ML uses gradients, not backprop) |
| Optimizer update (GD) | Some ML models | Yes | Only gradient-based models (e.g., logistic regression uses gradients; decision trees don't) |
| Batch-level updates | Sometimes | Very common | Mini-batch culture is DL |
| Epoch concept | Yes | Yes | For all iterative learners |


---

## Additional Notes

- The **cost function** computes the average of losses in a batch
- The **learning rate** controls how large each weight update is
- This describes **mini-batch gradient descent**, the most common approach in deep learning
-  **The optimizer updates weights after each batch**, not after each epoch

---

Hope yall understand, don’t forget to thank me 
