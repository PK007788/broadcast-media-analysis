# Mathematical Foundations & Formulas: Broadcast Media Analysis

This document provides a comprehensive, mathematically rigorous derivation and explanation of all formulas implemented in the **Broadcast Media Analysis: Rectangular Text Overlay Detection** project, cross-referenced with the codebase ([`train_baseline.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/train_baseline.py), [`calculate_iou.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/calculate_iou.py), [`train_dataset_adapted.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/train_dataset_adapted.py), and [`split_and_measure.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/split_and_measure.py)).

---

## 1. Intersection over Union (IoU) & Spatial Matching

### 1.1 Mathematical Definition
Intersection over Union (Jaccard Index) measures the spatial overlap between a predicted bounding box $B_{\text{pred}}$ and a ground-truth bounding box $B_{\text{gt}}$:

$$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$

Let the predicted box be parameterized by top-left and bottom-right corners:
$$B_{\text{pred}} = (x_1^p, y_1^p, x_2^p, y_2^p), \quad \text{with } x_2^p > x_1^p, \; y_2^p > y_1^p$$
and the ground-truth box:
$$B_{\text{gt}} = (x_1^g, y_1^g, x_2^g, y_2^g), \quad \text{with } x_2^g > x_1^g, \; y_2^g > y_1^g$$

#### Step 1: Intersection Rectangle Coordinates
The intersection box $(x_1^*, y_1^*, x_2^*, y_2^*)$ is formed by:
$$x_1^* = \max(x_1^p, x_1^g), \quad y_1^* = \max(y_1^p, y_1^g)$$
$$x_2^* = \min(x_2^p, x_2^g), \quad y_2^* = \min(y_2^p, y_2^g)$$

#### Step 2: Intersection Area
To prevent negative dimensions when boxes do not overlap ($x_2^* < x_1^*$ or $y_2^* < y_1^*$), dimensions are clamped at zero:
$$w_{\text{inter}} = \max(0, \, x_2^* - x_1^*)$$
$$h_{\text{inter}} = \max(0, \, y_2^* - y_1^*)$$
$$\text{Area}_{\text{inter}} = w_{\text{inter}} \times h_{\text{inter}}$$

#### Step 3: Union Area
By the Principle of Inclusion-Exclusion:
$$\text{Area}_{\text{union}} = \text{Area}(B_{\text{pred}}) + \text{Area}(B_{\text{gt}}) - \text{Area}_{\text{inter}}$$
where:
$$\text{Area}(B_{\text{pred}}) = (x_2^p - x_1^p) \times (y_2^p - y_1^p)$$
$$\text{Area}(B_{\text{gt}}) = (x_2^g - x_1^g) \times (y_2^g - y_1^g)$$

#### Step 4: Final Ratio
$$\text{IoU} = \frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{union}} + \epsilon}$$
where $\epsilon = 10^{-6}$ is a numerical stabilizer preventing division by zero.

### 1.2 Code Implementation Mapping
In [`train_baseline.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/train_baseline.py#L223-L235) and [`calculate_iou.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/calculate_iou.py#L49):
```python
def compute_iou(box1, box2):
    # Tensor broadcasting: box1 (N, 1, 4), box2 (1, M, 4) -> (N, M)
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)
```

---

## 2. SSDLite Multi-Task Objective Loss

### 2.1 Theoretical Formulation (Liu et al., ECCV 2016)
SSDLite trains using a multi-task loss that jointly optimizes confidence (classification) and localization (bounding box regression):

$$\mathcal{L}_{\text{total}}(x, c, l, g) = \frac{1}{N_{\text{pos}}} \left( \mathcal{L}_{\text{conf}}(x, c) + \alpha \mathcal{L}_{\text{loc}}(x, l, g) \right)$$

where:
- $N_{\text{pos}}$ is the number of matched positive default anchor boxes. If $N_{\text{pos}} = 0$, the loss is set to $0$.
- $\alpha = 1.0$ is the balancing hyperparameter between classification and localization.
- $x_{ij}^p \in \{0, 1\}$ is an indicator denoting whether default box $i$ is matched to ground truth box $j$ of class $p \in \{0, 1\}$ (where $0$ = background, $1$ = text overlay).

### 2.2 Confidence Loss with Online Hard Example Mining (OHEM)
In object detection, the background dominates the majority of anchor boxes (class imbalance ratio $> 100:1$). To address this, SSDLite uses **Hard Negative Mining**:
1. Compute negative loss for all unmatched default boxes.
2. Sort negative anchors by loss in descending order.
3. Retain only the top negative anchors such that:
$$N_{\text{neg}} = 3 \times N_{\text{pos}} \quad (\text{ratio } 3:1)$$

The confidence loss is the Cross-Entropy loss over positive anchors and mined hard negative anchors:
$$\mathcal{L}_{\text{conf}}(x, c) = -\sum_{i \in \text{Pos}}^{N_{\text{pos}}} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in \text{HardNeg}} \log(\hat{c}_i^0)$$
where softmax class probabilities are:
$$\hat{c}_i^p = \frac{\exp(c_i^p)}{\sum_{k=0}^{C-1} \exp(c_i^k)}$$
For our binary detector ($C=2$), this evaluates class $1$ (text overlay) against class $0$ (background).

---

## 3. Smooth L1 Bounding Box Regression Loss

### 3.1 Bounding Box Parameterization (Offsets)
Rather than predicting absolute pixel coordinates $(x_1, y_1, x_2, y_2)$, the network predicts regression offsets relative to default anchor box $d = (d^{cx}, d^{cy}, d^w, d^h)$:

$$\hat{g}_j^{cx} = \frac{g_j^{cx} - d_i^{cx}}{d_i^w}, \quad \hat{g}_j^{cy} = \frac{g_j^{cy} - d_i^{cy}}{d_i^h}$$
$$\hat{g}_j^w = \log\left(\frac{g_j^w}{d_i^w}\right), \quad \hat{g}_j^h = \log\left(\frac{g_j^h}{d_i^h}\right)$$

where:
- $(g_j^{cx}, g_j^{cy}, g_j^w, g_j^h)$ are center coordinates, width, and height of ground truth box $j$.
- $(d_i^{cx}, d_i^{cy}, d_i^w, d_i^h)$ are center coordinates, width, and height of default anchor $i$.
- $l_i = (l_i^{cx}, l_i^{cy}, l_i^w, l_i^h)$ is the model's predicted offset vector.

### 3.2 Smooth L1 (Huber) Loss Function
The localization loss is defined over matched positive anchors:
$$\mathcal{L}_{\text{loc}}(x, l, g) = \sum_{i \in \text{Pos}} \sum_{m \in \{cx, cy, w, h\}} \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)$$
where:
$$\text{smooth}_{L1}(u) = \begin{cases} 
0.5 u^2 & \text{if } |u| < 1 \\ 
|u| - 0.5 & \text{otherwise} 
\end{cases}$$

#### Why Smooth L1?
- When $|u| \ge 1$ (large initial errors), the derivative is constant ($\pm 1$), preventing exploding gradients.
- When $|u| < 1$ (fine alignment), the loss becomes quadratic ($0.5 u^2$), smoothly vanishing to zero and preventing oscillation around the minimum.

---

## 4. Anchor Box Scale & Aspect Ratio Generation

### 4.1 Multi-Scale Feature Pyramid Scales
SSDLite320 extracts predictions across $m = 6$ feature maps of decreasing spatial resolution ($19\times19$, $10\times10$, $5\times5$, $3\times3$, $2\times2$, $1\times1$).

The scale $s_k$ of anchors for feature map $k \in \{1, \dots, m\}$ is determined by linear interpolation:
$$s_k = s_{\min} + \frac{s_{\max} - s_{\min}}{m - 1}(k - 1)$$
In [`split_and_measure.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/split_and_measure.py#L131-L134) and [`train_dataset_adapted.py`](file:///c:/Users/prajn/OneDrive/Desktop/iitp/train_dataset_adapted.py#L88):
$$s_{\min} = 0.20 \quad (\text{min\_ratio} = 20), \quad s_{\max} = 0.95 \quad (\text{max\_ratio} = 95)$$

### 4.2 Aspect Ratio Transformations
For a given scale $s_k$ and aspect ratio $a_r$:
$$w_k^a = s_k \sqrt{a_r}, \quad h_k^a = \frac{s_k}{\sqrt{a_r}}$$

#### Area Preservation Invariant:
$$\text{Area} = w_k^a \times h_k^a = \left(s_k \sqrt{a_r}\right) \times \left(\frac{s_k}{\sqrt{a_r}}\right) = s_k^2$$
#### Aspect Ratio Invariant:
$$\frac{w_k^a}{h_k^a} = \frac{s_k \sqrt{a_r}}{s_k / \sqrt{a_r}} = a_r$$

### 4.3 Baseline vs Dataset-Adapted Configurations
- **Baseline (Model A)**: Default TorchVision configuration:
  $$a_r \in \{2, 3\} \quad \text{yielding 6 anchors per location (including reciprocals } 1/2, 1/3 \text{ and scale step)}$$
- **Dataset-Adapted (Model B)**: Wide aspect ratios tailored for broadcast news tickers and lower-thirds:
  $$a_r \in \{2, 3, 5, 8\} \quad \text{yielding 10 anchors per location}$$
  $$\text{Aspect ratio 8: } w = s_k \sqrt{8} \approx 2.83 s_k, \quad h = \frac{s_k}{\sqrt{8}} \approx 0.35 s_k$$
  This allows anchors to match wide banners that span $>80\%$ of the screen width with small vertical thickness ($<15\%$).

---

## 5. Evaluation Metrics Formulations

### 5.1 Mean IoU (mIoU)
Evaluated on true positive overlay detections ($\text{IoU} \ge 0.5$):
$$\text{mIoU} = \frac{1}{|M|} \sum_{(p, g) \in M} \text{IoU}(B_p, B_g)$$
- **Baseline Model A**: $\mathbf{0.7477}$ (74.77% mean overlap on test set).
- **Adapted Model B**: $\mathbf{0.6899}$ (68.99% mean overlap on test set).

### 5.2 Precision, Recall, and F1-Score
Given True Positives ($TP = 31$), False Positives ($FP = 4$), False Negatives ($FN = 1$):

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{31}{31 + 4} = \frac{31}{35} \approx \mathbf{0.8857} \; (88.57\%)$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{31}{31 + 1} = \frac{31}{32} \approx \mathbf{0.9688} \; (96.88\%)$$

$$\text{F}_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.8857 \times 0.9688}{0.8857 + 0.9688} \approx \mathbf{0.9254}$$

### 5.3 Computational Complexity & FLOPs
Measured via `fvcore.nn.FlopCountAnalysis`:
- **FLOPs (Floating Point Operations)**:
  $$\text{FLOPs}_{\text{Conv2D}} = 2 \times H_{\text{out}} \times W_{\text{out}} \times (K_h \times K_w \times C_{\text{in}}) \times C_{\text{out}}$$
  $$\text{Depthwise Conv: } 2 \times H_{\text{out}} \times W_{\text{out}} \times (K_h \times K_w \times 1) \times C_{\text{in}}$$
  $$\text{Pointwise Conv (1x1): } 2 \times H_{\text{out}} \times W_{\text{out}} \times (1 \times 1 \times C_{\text{in}}) \times C_{\text{out}}$$
  SSDLite reduces standard convolution compute by $\approx \frac{1}{K^2} + \frac{1}{C_{\text{out}}} \approx \frac{1}{9} + \text{small} \approx 85\text{--}90\%$ reduction.
- **Total Compute**: $0.427 \times 10^9$ FLOPs ($0.427$ GFLOPs), executing in **$32.69$ ms (30.6 FPS)** on standard CPU.

---

## 6. Summary of Mathematical Rigor

| Concept | Mathematical Equation | Purpose |
| :--- | :--- | :--- |
| **IoU** | $\frac{\max(0, x_2^* - x_1^*) \cdot \max(0, y_2^* - y_1^*)}{\text{Area}_1 + \text{Area}_2 - \text{Area}_{\text{inter}}}$ | Strict spatial overlap and NMS matching |
| **Total Loss** | $\frac{1}{N_{\text{pos}}} (\mathcal{L}_{\text{conf}} + \alpha \mathcal{L}_{\text{loc}})$ | Multi-task joint optimization |
| **OHEM Loss** | $-\sum_{i \in \text{Pos}} \log(\hat{c}_i^p) - \sum_{i \in \text{HardNeg}} \log(\hat{c}_i^0)$ | Mitigates class imbalance with 3:1 ratio |
| **Smooth L1** | $\begin{cases} 0.5 u^2 & \|u\| < 1 \\ \|u\| - 0.5 & \text{otherwise} \end{cases}$ | Robust gradient descent on box coordinate offsets |
| **Anchor Scale** | $s_k = s_{\min} + \frac{s_{\max} - s_{\min}}{m-1}(k-1)$ | Pyramid scale progression across 6 levels |
| **Aspect Ratios** | $w = s_k \sqrt{a_r}, \; h = s_k / \sqrt{a_r}$ | Shape coverage for wide horizontal overlays |

All formulas have been verified against the official PyTorch TorchVision SSD/SSDLite architecture implementation and the project source code.
