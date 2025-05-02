# Cấu trúc của CustomStackingClassifier 
```
class_name: CustomStackingClassifier
estimators:
    - LogisticRegression(C = 0.1)
    - GaussianNB(var_smoothing=1e-8)
    - SGDClassifier(alpha=10, loss='log_loss')
final_estimator: LogisticRegression(C = 0.1)
```

