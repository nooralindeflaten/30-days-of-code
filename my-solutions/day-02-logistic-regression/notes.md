# Solution notes
I'm not 100% happy with the current code it could be cleaner, but the functionality satisfies the challenge implementations. 
- The confusion matrix has a small difference to sklearn. My matrix is set-up in a different order than sklearn, but the rest of my code is set-up to compute the metrics with this in mind. 

## Results
After running the script this is the output: 

Blobs results: 

- Confusion Matrix:
 
    [[132   3]
    [ 18 147]]

- Accuracy: 0.93
- Precision: 0.9777777777777777
- Recall: 0.88
- F1 Score: 0.9263157894736842

Breast Cancer results: 

Confusion Matrix Mine:
 [[355   8]
 [  2 204]]

Confusion Matrix Sklearn:
 [[207   5]
 [  2 355]]

- Accuracy: 0.9824253075571178
- Accuracy Sklearn: 0.9876977152899824
- Precision: 0.977961432506887
- Precision Sklearn: 0.9861111111111112
- Recall: 0.9943977591036415
- Recall Sklearn: 0.9943977591036415
- F1 Score: 0.9861111111111112
- F1 Score Sklearn: 0.9902370990237099

## AI-usage
1. For the breastcancer dataset I wanted to make sure using StandardScaler().fit_transform(X) was the right call as this is what I've used previously.

2. I asked Claude to show me a sample of a "test script", but I didn't use this, but I wanted to know if the data needed a test-train split in the models. Apart from that the code Claude provided did the same functionality as mine except for one function. It suggested using a check-function for validation rather than just by eyesight. I didn't implement it as I felt too much of my code would have to be changes, and for me it seemed like too much work/changing needed to be done to fit this function. 

```python
def check(name, mine, theirs, tol=1e-6):
    ok = np.isclose(mine, theirs, atol=tol)
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:<10} mine={mine:.4f}  sklearn={theirs:.4f}")
    return ok
 
 
def run_metrics_check(y_true, y_pred, label=""):
    print(f"\n--- Metrics check{(' (' + label + ')') if label else ''} ---")
    all_ok = True
 
    my_cm = confusion_matrix(y_true, y_pred)
    sk_cm = skm.confusion_matrix(y_true, y_pred)
    cm_ok = np.array_equal(my_cm, sk_cm)
    print(f"  [{'OK ' if cm_ok else 'FAIL'}] confusion_matrix")
    print(f"      mine:\n{my_cm}")
    print(f"      sklearn:\n{sk_cm}")
    all_ok &= cm_ok
 
    all_ok &= check("accuracy", accuracy(y_true, y_pred), skm.accuracy_score(y_true, y_pred))
    all_ok &= check("precision", precision(y_true, y_pred), skm.precision_score(y_true, y_pred))
    all_ok &= check("recall", recall(y_true, y_pred), skm.recall_score(y_true, y_pred))
    all_ok &= check("f1", f1_score(y_true, y_pred), skm.f1_score(y_true, y_pred))
    return all_ok
```
3. I googled clipping for the sigmoid function to see what numerical bounds should be set. 

## Math


**Sigmoid**

&emsp;$\sigma(x)=\frac{1}{1+e^{-x}}$

**Binary cross-entropy loss**

Source: https://stackoverflow.com/questions/67615051/implementing-binary-cross-entropy-loss-gives-different-answer-than-tensorflows

&emsp;$\text{BCE}=-\frac{1}{N}\sum _{i=1}^{N}\left[y_{i}\log (\^{y}_{i})+(1-y_{i})\log (1-\^{y}_{i})\right]$


**Logistic Regression**
The full class follows the structure of the mathematical functions. Didn't need AI here because I had it in school. 

Source: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/

X has been turned into a matrix

&emsp;$X = \begin{bmatrix} x_{11}  & ... & x_{1m}\\ x_{21}  & ... & x_{2m} \\  \vdots & \ddots  & \vdots  \\ x_{n1}  & ... & x_{nm} \end{bmatrix}$

Then we use the sigmoid function on this z-value

&emsp;$z = w\cdot X +b$

**Metrics**
All the metrics formulas were retrieved from sklearn's user-documentation guide to get the exact math. 

For example accuracy score:

&emsp; $\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$