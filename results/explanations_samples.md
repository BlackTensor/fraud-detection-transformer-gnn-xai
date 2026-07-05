# SHAP + Ollama Spot-Check (Task 6.3)

Pipeline: `best_model.pt` (Transformer + GAT) -> SHAP KernelExplainer (input-feature level) -> local Ollama `llama3.2:3b` NL explanation.

For each transaction we list the top-5 signed SHAP factors and the LLM paragraph. Two objective checks per row:
- **Additive check**: `base + sum(all 32 SHAP)` should equal the model's predicted probability (SHAP is exactly additive here).
- **Sign-consistent**: the net SHAP push direction matches `sign(prob - base)`.

Frauds are the most-confident true frauds; legits the most-confident true legits (so the explanations are clear, correctly-classified examples).

## 1. FRAUD — TransactionID 3529001 (true isFraud=1)

- Predicted fraud probability: **75.4%**
- Raw: amount=54.225, hour=13, merchant(ProductCD)=C, device=mobile, region(addr1)=299.0
- base=0.149 | additive check=0.754 | net direction=**raise** | sign-consistent=**True**

Top-5 SHAP factors (+ raises risk, - lowers):

| rank | feature | model_value | SHAP |
|---|---|---|---|
| 1 | C1 | 0.446 | +0.222 |
| 2 | C11 | 0.381 | +0.121 |
| 3 | C8 | 0.577 | +0.084 |
| 4 | C12 | 0.459 | +0.063 |
| 5 | DeviceType | 0.100 | +0.033 |

NL explanation (`ollama:llama3.2:3b`):

> The transaction has been flagged as high-risk due to several contributing factors. Factor C1 raised risk by +0.222, while factor C11 increased risk by +0.121. Additionally, factor DeviceType had a moderate impact, raising risk by +0.033. These factors outweighed the others, indicating a higher likelihood of fraud.

## 2. FRAUD — TransactionID 3540438 (true isFraud=1)

- Predicted fraud probability: **75.3%**
- Raw: amount=54.225, hour=4, merchant(ProductCD)=C, device=mobile, region(addr1)=299.0
- base=0.192 | additive check=0.753 | net direction=**raise** | sign-consistent=**True**

Top-5 SHAP factors (+ raises risk, - lowers):

| rank | feature | model_value | SHAP |
|---|---|---|---|
| 1 | C1 | 0.452 | +0.202 |
| 2 | C11 | 0.390 | +0.108 |
| 3 | C8 | 0.577 | +0.076 |
| 4 | C12 | 0.468 | +0.060 |
| 5 | DeviceType | 0.100 | +0.035 |

NL explanation (`ollama:llama3.2:3b`):

> The transaction has been flagged as high-risk due to several contributing factors. Factor C1 raised risk by +0.202, while factor C11 increased risk by +0.108. Additionally, the device type (mobile) and factor C8 both raised risk by smaller but still significant amounts (+0.035 and +0.076, respectively). These findings suggest that the transaction warrants further review and potential denial.

## 3. FRAUD — TransactionID 3528996 (true isFraud=1)

- Predicted fraud probability: **75.2%**
- Raw: amount=54.225, hour=13, merchant(ProductCD)=C, device=mobile, region(addr1)=299.0
- base=0.163 | additive check=0.752 | net direction=**raise** | sign-consistent=**True**

Top-5 SHAP factors (+ raises risk, - lowers):

| rank | feature | model_value | SHAP |
|---|---|---|---|
| 1 | C1 | 0.433 | +0.198 |
| 2 | C11 | 0.372 | +0.111 |
| 3 | C8 | 0.577 | +0.080 |
| 4 | C12 | 0.449 | +0.070 |
| 5 | DeviceType | 0.100 | +0.033 |

NL explanation (`ollama:llama3.2:3b`):

> The transaction is flagged as high-risk due to several contributing factors. Factor C1 raised risk by +0.198, while factor C11 increased risk by +0.111. Additionally, factor DeviceType had a moderate impact, raising risk by +0.033. These factors outweighed the others, indicating a higher likelihood of fraud.

## 4. LEGIT — TransactionID 3574782 (true isFraud=0)

- Predicted fraud probability: **5.2%**
- Raw: amount=225.0, hour=23, merchant(ProductCD)=H, device=desktop, region(addr1)=325.0
- base=0.105 | additive check=0.052 | net direction=**lower** | sign-consistent=**True**

Top-5 SHAP factors (+ raises risk, - lowers):

| rank | feature | model_value | SHAP |
|---|---|---|---|
| 1 | C3 | 13.067 | -0.074 |
| 2 | DeviceType | 0.160 | +0.013 |
| 3 | C14 | -0.160 | +0.011 |
| 4 | log_amt | 1.109 | +0.011 |
| 5 | C9 | -0.261 | -0.007 |

NL explanation (`ollama:llama3.2:3b`):

> The transaction is considered low-risk due to several factors that lower the risk. Factor C9 lowered risk by +0.007, while factor C3 lowered risk by -0.074. Despite some factors raising risk, such as device type and log amount, their impact was not significant enough to outweigh the lowering effects of these two factors. Overall, the model predicts a 5% chance of fraud for this transaction.

## 5. LEGIT — TransactionID 3575703 (true isFraud=0)

- Predicted fraud probability: **5.2%**
- Raw: amount=100.0, hour=13, merchant(ProductCD)=H, device=desktop, region(addr1)=325.0
- base=0.103 | additive check=0.052 | net direction=**lower** | sign-consistent=**True**

Top-5 SHAP factors (+ raises risk, - lowers):

| rank | feature | model_value | SHAP |
|---|---|---|---|
| 1 | C3 | 13.067 | -0.058 |
| 2 | C14 | -0.160 | +0.008 |
| 3 | DeviceType | 0.160 | +0.007 |
| 4 | C1 | -0.090 | -0.006 |
| 5 | C5 | -0.213 | +0.004 |

NL explanation (`ollama:llama3.2:3b`):

> The transaction is considered low-risk due to several factors that lower the risk. Factor C1 lowered risk by -0.006, and factor C3 lowered risk by -0.058. While some factors, such as DeviceType (+0.007) and factor C5 (+0.004), raised risk, their impact was relatively minor compared to the significant reductions from factor C1 and C3. Overall, the model predicts a 5% fraud rate for this transaction.
