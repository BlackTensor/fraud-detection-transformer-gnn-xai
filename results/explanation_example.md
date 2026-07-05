# NL Explanation — best model (Transformer + GAT)

- TransactionID: 3529001 (true label isFraud=1)
- Predicted fraud probability: 75.4%
- Explanation source: `ollama:llama3.2:3b`

## Prompt sent to the local LLM
```
You are a fraud-analysis assistant. Explain the model's decision for a bank
officer using ONLY the evidence listed below.

STRICT RULES — follow all of them:
- Use ONLY the numbers and fields given here. Do NOT add outside knowledge,
  history, statistics, or any context that is not literally listed.
- The factor codes (e.g. C1, C8, C11) are ANONYMIZED features. Do NOT guess,
  state, or imply what they represent.
- Refer to each factor ONLY by its exact code/name as written. NEVER describe a
  C... or D... factor as a device, email, merchant, region, time, OS, or browser
  feature — those codes are anonymized and unnamed.
- Each factor's 'value=' is the model's internal scaled input, NOT a
  probability or percentage — do not convert it to a percent or reinterpret it.
- Describe each factor only as raising or lowering risk by its SHAP number.
- Do NOT invent claims about timing, location, device reputation, or merchant
  behaviour beyond what is explicitly written.

Prediction: 75% fraud
Transaction: amount=54.225, hour=13, merchant=C,
             device=mobile (SM-G920P Build/NRD90M), region=299.0
Top contributing factors (SHAP, + raises risk, - lowers it):
  - factor C1: +0.222 (raises risk)  [model input value=0.446]
  - factor C11: +0.121 (raises risk)  [model input value=0.381]
  - factor C8: +0.084 (raises risk)  [model input value=0.577]
  - factor C12: +0.063 (raises risk)  [model input value=0.459]
  - factor DeviceType: +0.033 (raises risk)  [model input value=0.100]

Write 3-4 plain-English sentences a bank officer could read. Refer to each
factor as 'factor <CODE>' using its EXACT code and its EXACT signed SHAP value
from the list above (never the 'value=' number) — e.g. phrased like "factor Z9
raised risk by +0.21" (substitute the real codes and their real SHAP values).
Note that the risk-RAISING factors outweighed the rest, so the transaction looks high-risk.
```

## Generated explanation

The transaction has been flagged as high-risk due to several factors. Factor C1 raised risk by +0.222, while factor C11 increased risk by +0.121. Additionally, factor C8 and C12 also contributed to the elevated risk, with values of +0.084 and +0.063, respectively. These factors outweighed the relatively minor impact of device type, which raised risk by +0.033.
