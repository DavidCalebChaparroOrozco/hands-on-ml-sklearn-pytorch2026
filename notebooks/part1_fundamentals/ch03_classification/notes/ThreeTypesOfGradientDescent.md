# The three types of gradient descent

1. Batch
2. Stochastic
3. Mini-batch

> Same algorithm, three ways to walk it.

---

## Finding the Best Line
We use a **linear regression** problem: as **x** increases, the **y** we are predicting also increases.

### Example:
More $square^2$ in a house → higher **price** we are predicting.

> I can draw **n different lines** through the same points.

But **only one** minimizes the prediction error. Finding it is the goal.

---

## Each line has its own error
Every possible line (every slope value) produces different predictions, and therefore a different error.

To compare them, we calculate the **Mean Squared Error (MSE)** for each slope. As the slope gets closer to the optimal value, the MSE decreases. Once we move past the optimal slope, the MSE starts increasing again.

Example
| Slope | MSE  |
| ----- | ---- |
| 0.80  | 2.15 |
| 0.95  | 0.78 |
| 1.06  | 0.35 |
| 1.20  | 0.92 |
| 1.35  | 2.47 |

In this example, a slope of 1.06 produces the lowest MSE (0.35), making it the best candidate among the tested values.

![alt text](../images/EachLineHasItsOwnError.png)
> ### **MSE by slope**
> If we plot the MSE against the slope values, we obtain a **U-shaped curve**. The lowest point of this curve represents the optimal slope—the value that minimizes the prediction error.