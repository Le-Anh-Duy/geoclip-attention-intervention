# Ghi chú ý tưởng thesis: Grounding vocabulary và attention intervention cho GeoCLIP

## 1. Bài toán tổng quát

Pipeline nghiên cứu hiện tại:

```text
Candidate prompt/object vocabulary
        ↓
Text encoder của Grounding DINO
        ↓
Embedding bank đã đánh ID
        ↓
Chọn một subset embedding rời rạc
        ↓
Grounding DINO sinh region / bounding box
        ↓
Can thiệp attention hoặc representation của GeoCLIP
        ↓
Đánh giá geolocation và explanation faithfulness
```

Mục tiêu tổng quát:

\[
S^* = \arg\max_{S \subseteq \{1,\ldots,M\},\, |S|\le K} J(S)
\]

Trong đó:

- \(M\): số prompt/object embedding ứng viên.
- \(S\): subset ID embedding được chọn.
- \(K\): ngân sách tối đa về số embedding.
- \(J(S)\): objective downstream sau Grounding DINO và GeoCLIP intervention.

Đây là bài toán **combinatorial subset selection**, không phải continuous prompt interpolation.

---

## 2. Hai hướng xây dựng vocabulary ban đầu

### Hướng A — Controlled empirical study

Thử nghiệm có hệ thống nhiều loại prompt:

- Object names: `road sign`, `building`, `vehicle`, `utility pole`.
- Synonyms: `road sign`, `traffic sign`, `street sign`.
- Attribute phrases: `concrete utility pole`, `tropical vegetation`.
- Geographic clue phrases: `country-specific road sign`.
- Contextual phrases: `objects useful for identifying the country`.

Các yếu tố nên ablate:

- Prompt ngắn và prompt dài.
- Singular và plural.
- Concrete object và abstract concept.
- Số synonym.
- Thứ tự prompt.
- Số category trong cùng caption.
- Prompt riêng lẻ và prompt ensemble.

Mục tiêu của empirical study không chỉ là tìm prompt tốt nhất, mà còn phân tích:

- Prompt nào tạo coverage tốt.
- Prompt nào tạo box quá lớn.
- Prompt nào không ổn định.
- Prompt nào tạo nhiều duplicate boxes.
- Prompt tốt cho detection có thực sự tốt cho downstream intervention hay không.

### Hướng B — Validation-guided vocabulary discovery

Tạo một candidate pool lớn rồi tự động tìm subset có điểm downstream tốt nhất.

Nguồn tạo candidates:

- Taxonomy thủ công về geographic clues.
- Noun phrases từ caption model.
- Object/tagging model.
- LLM sinh synonym, subcategory và attribute.
- Label từ các dataset liên quan.
- Các region description có sẵn.

Sau đó:

1. Embed từng candidate.
2. Gán ID cho embedding.
3. Đánh giá từng embedding riêng.
4. Loại prompt yếu hoặc dư thừa.
5. Search subset trên tập còn lại.

---

## 3. Candidate taxonomy gợi ý

### Transportation

- road
- lane marking
- vehicle
- motorcycle
- bus
- license plate
- traffic light
- traffic sign
- road sign
- bollard

### Infrastructure

- utility pole
- electrical wire
- street lamp
- sidewalk
- curb
- guardrail
- bridge
- bus stop
- road barrier

### Architecture

- building
- house
- facade
- roof
- window
- balcony
- storefront
- shop sign
- building material
- religious building

### Natural environment

- vegetation
- tree
- tropical vegetation
- mountain
- hill
- soil
- coastline
- river
- agricultural field

### Textual clues

- written text
- storefront sign
- road sign
- directional sign
- banner
- advertisement board
- license plate

---

## 4. Cách đưa subset embedding vào Grounding DINO

Với subset:

\[
S = \{i_1,i_2,\ldots,i_k\}
\]

ta lấy:

\[
Q_S = [e_{i_1},e_{i_2},\ldots,e_{i_k}]
\]

Grounding DINO tạo region cho từng embedding:

\[
R_S(x) = \operatorname{Merge}
\left(
R_{i_1}(x),R_{i_2}(x),\ldots,R_{i_k}(x)
\right)
\]

Cần cố định rõ cơ chế `Merge`:

- Class-agnostic NMS.
- NMS riêng theo prompt.
- Union mask.
- Top-\(B\) box toàn cục.
- Tối đa \(b\) box cho mỗi embedding.
- Ngưỡng confidence cố định.
- Cách xử lý box trùng nhau giữa nhiều prompt.

Không nên thay đổi cách merge theo từng subset vì sẽ làm objective khó diễn giải.

---

## 5. Unary screening trước khi search subset

Đánh giá từng candidate riêng lẻ.

### Coverage

\[
C(i)=\frac{1}{N}\sum_x
\mathbf{1}\left(|R_i(x)|>0\right)
\]

### Area appropriateness

Đo tỷ lệ diện tích box trên toàn ảnh. Prompt tạo box gần toàn ảnh thường ít phù hợp cho region intervention.

### Stability

Đo độ ổn định của region qua resize, crop nhẹ hoặc augmentation:

\[
S(i)=\operatorname{IoU}
\left(
R_i(x), T^{-1}R_i(T(x))
\right)
\]

### Causal utility

Đo thay đổi output GeoCLIP khi can thiệp vào region:

\[
U(i)=
\frac{1}{N}
\sum_x
\left|
s(x)-s_{\operatorname{intervene}(R_i(x))}
\right|
\]

Nếu có ground-truth GPS, có thể dùng:

- Thay đổi similarity với GPS đúng.
- Thay đổi rank của GPS đúng.
- Thay đổi geodesic error.
- Thay đổi Recall@K.

### Selectivity so với random region

\[
Q(i)=
\Delta s_{\text{concept region}}
-
\mathbb{E}
\left[
\Delta s_{\text{random area-matched region}}
\right]
\]

Điều này tránh việc một prompt được đánh giá cao chỉ vì nó sinh box lớn.

---

## 6. Objective chọn subset

Không nên chỉ dùng accuracy cuối cùng.

Một objective tổng quát:

\[
J(S)=
J_{\text{geo}}(S)
+\lambda_f J_{\text{faith}}(S)
+\lambda_s J_{\text{stability}}(S)
-\lambda_r J_{\text{redundancy}}(S)
-\lambda_b J_{\text{box-cost}}(S)
\]

### Geolocation utility

Có thể dùng:

- Negative median geodesic error.
- Mean geodesic error.
- Recall@1, Recall@5, Recall@10.
- Rank của GPS đúng.
- Similarity margin giữa GPS đúng và hard negative.

### Faithfulness

\[
J_{\text{faith}}=
\Delta s_{\text{selected regions}}
-
\Delta s_{\text{random area-matched regions}}
\]

### Stability

Độ nhất quán của region và hiệu ứng intervention qua augmentation.

### Redundancy

Hai prompt có thể gần như tạo cùng region.

Có thể đo redundancy bằng:

- Cosine similarity giữa text embeddings.
- IoU trung bình giữa detection boxes.
- Correlation giữa intervention effects.
- Tỷ lệ hai prompt cùng phát hiện một region.

Ví dụ:

\[
J_{\text{redundancy}}(S)
=
\frac{2}{|S|(|S|-1)}
\sum_{i<j}\rho_{ij}
\]

### Region cost

Có thể phạt:

- Số box trung bình mỗi ảnh.
- Tổng area ratio.
- Chi phí inference.
- Số region sau NMS.

---

## 7. Cardinality budget và region budget

### Cardinality-constrained

\[
|S|\le K
\]

Chọn tối đa \(K\) embeddings.

### Region-budget constrained

\[
\frac{1}{N}\sum_x |R_S(x)| \le B
\]

Dạng này công bằng hơn vì hai subset có cùng số prompt chưa chắc tạo cùng số region.

Nên báo cáo cả hai:

- Fixed vocabulary size.
- Fixed number of regions hoặc fixed inference budget.

---

## 8. Baseline bắt buộc

### Random subset

Chọn ngẫu nhiên \(K\) embeddings, chạy nhiều seed.

### Top-\(K\) theo detection confidence

Chọn prompt có confidence Grounding DINO cao nhất.

### Top-\(K\) theo unary downstream score

\[
u_i = J(\{i\}) - J(\varnothing)
\]

### Top-\(K\) theo diversity

Chọn embeddings phân tán trong text embedding space.

### Manual geographic vocabulary

Tập vocabulary do con người thiết kế.

### Greedy downstream selection

Chọn theo marginal downstream gain.

Các baseline này giúp chứng minh rằng việc search dựa trên downstream intervention tốt hơn chỉ dựa vào detection confidence hoặc intuition.

---

## 9. Greedy forward selection

Khởi tạo:

\[
S_0=\varnothing
\]

Tại bước \(t\):

\[
i^*=
\arg\max_{i\notin S_t}
\left[
J(S_t\cup\{i\})-J(S_t)
\right]
\]

Sau đó:

\[
S_{t+1}=S_t\cup\{i^*\}
\]

Điều kiện dừng:

- Đã đạt \(K\).
- Marginal gain nhỏ hơn \(\epsilon\).
- Vượt region budget.
- Objective đã bão hòa.

Ưu điểm:

- Training-free.
- Dễ triển khai.
- Dễ giải thích.
- Tạo được thứ tự prompt được chọn.
- Dễ vẽ curve \(K \mapsto J(S_K)\).

---

## 10. Refinement sau greedy

### Backward deletion

Thử loại từng phần tử khỏi subset:

\[
S' = S\setminus\{i\}
\]

Giữ phép xóa nếu objective không giảm hoặc tăng.

### One-swap local search

Thử thay:

\[
S'=(S\setminus\{i\})\cup\{j\}
\]

với \(i\in S\), \(j\notin S\).

Lặp đến khi không còn swap cải thiện objective.

### Beam search

Giữ top-\(B\) subset ở mỗi độ sâu. Hữu ích khi có interaction mạnh giữa prompt.

### Evolutionary search

Biểu diễn subset bằng binary mask:

- Mutation: thêm, xóa hoặc thay prompt.
- Crossover: kết hợp hai subset.
- Fitness: downstream objective.

Phù hợp khi candidate pool tương đối lớn.

---

## 11. Pairwise interaction và synergy

Hai prompt có thể bổ trợ hoặc dư thừa.

\[
I_{ij}
=
J(\{i,j\})
-J(\{i\})
-J(\{j\})
+J(\varnothing)
\]

- \(I_{ij}>0\): synergy.
- \(I_{ij}<0\): redundancy hoặc interference.
- \(I_{ij}\approx0\): gần độc lập.

Không cần tính toàn bộ \(M^2\). Có thể:

1. Unary screening.
2. Giữ top 20–50 candidates.
3. Chỉ tính pairwise interaction trong nhóm này.
4. Dùng interaction để hỗ trợ clustering hoặc diversity-aware selection.

---

## 12. Prompt ensembling khác subset selection

Cần phân biệt:

### Subset selection

Chọn nhiều embedding riêng:

\[
Q_S=[e_{i_1},\ldots,e_{i_k}]
\]

Mỗi embedding vẫn là một query độc lập.

### Weighted prompt ensemble

Nhiều prompt đại diện cho cùng một concept, sau đó hợp nhất score:

\[
s(b,c)=\sum_m w_m s(b,p_m)
\]

### Continuous embedding interpolation

\[
q=\sum_i w_i e_i
\]

Hiện tại hướng chính là **subset embedding rời rạc**, không phải continuous interpolation.

Continuous interpolation có thể giữ làm ablation nâng cao sau này.

---

## 13. Search tham số intervention

Sau khi chọn được vocabulary subset, tiếp tục search các tham số intervention:

- Layer.
- Head.
- Region.
- Intervention strength \(\alpha\).
- Number of consecutive layers.
- Threshold.
- Top-\(k\) patches hoặc heads.
- Intervention type: suppress, amplify, replace, mask.

Pipeline gợi ý:

```text
Causal sensitivity scan
        ↓
Shortlist layer / head / region
        ↓
Search intervention strength và interval
        ↓
Backward hoặc local refinement
        ↓
Held-out evaluation
```

Không nên search toàn bộ vocabulary và toàn bộ intervention parameters cùng lúc ngay từ đầu vì search space sẽ quá lớn.

---

## 14. Multi-stage search được khuyến nghị

### Stage A — Vocabulary candidate generation

Tạo khoảng 100–300 candidates.

### Stage B — Unary screening

Giữ khoảng 30–80 candidates dựa trên:

- Coverage.
- Area.
- Stability.
- Causal utility.
- Selectivity.
- Redundancy.

### Stage C — Vocabulary subset search

Dùng:

1. Greedy forward.
2. Backward cleanup.
3. One-swap refinement.

### Stage D — Intervention parameter search

Search trên subset đã chọn:

- Layer interval.
- Head group.
- Strength.
- Threshold.
- Region budget.

### Stage E — Joint local refinement

Cho phép một số thay đổi nhỏ đồng thời:

- Đổi một prompt.
- Đổi strength.
- Đổi layer interval.

---

## 15. Multi-fidelity evaluation

Nếu mỗi evaluation tốn nhiều thời gian, dùng successive halving hoặc Hyperband.

Ví dụ:

| Mức | Số ảnh | GPS gallery | Augmentation |
|---|---:|---:|---:|
| Thấp | 200 | nhỏ | 1 |
| Trung bình | 1.000 | trung bình | 4 |
| Cao | toàn validation | đầy đủ | 8 |

Nhiều cấu hình được chạy ở fidelity thấp. Chỉ các cấu hình tốt mới được đánh giá đầy đủ.

---

## 16. Phân chia dữ liệu

Không nên search vocabulary trên cùng tập dùng để báo cáo kết quả cuối.

Thiết kế gợi ý:

```text
Prompt-development split:
- Sinh candidate.
- Unary screening.
- Search vocabulary.

Intervention-validation split:
- Search layer, head, strength và threshold.

Held-out test split:
- Chỉ đánh giá cuối.
```

Hoặc nested validation nếu dữ liệu đủ lớn.

Nên kiểm tra transfer:

- Search vocabulary trên một vùng, test trên vùng khác.
- Search trên một checkpoint Grounding DINO, test trên checkpoint khác.
- Search trên một GeoCLIP backbone, test trên backbone khác.

---

## 17. Các research question có thể dùng

### RQ1

Loại prompt nào tạo region hữu ích nhất cho GeoCLIP attention intervention?

### RQ2

Vocabulary được validation-guided search có tốt hơn vocabulary thủ công không?

### RQ3

Subset selection dựa trên downstream causal utility có tốt hơn selection dựa trên Grounding DINO confidence không?

### RQ4

Diversity và redundancy giữa prompt ảnh hưởng thế nào tới subset tối ưu?

### RQ5

Vocabulary được search có generalize sang vùng địa lý, dataset hoặc backbone khác không?

### RQ6

Có tồn tại một vocabulary subset nhỏ nhưng vẫn giữ phần lớn hiệu quả của tập vocabulary lớn không?

---

## 18. Framing đóng góp luận văn

Một framing có thể dùng:

> **Validation-guided grounding vocabulary discovery for attention intervention in image geolocation.**

Hoặc:

> **Training-free discrete prompt subset search for grounded and explainable GeoCLIP intervention.**

Đóng góp tiềm năng:

1. Xây dựng candidate vocabulary bank cho geographic visual clues.
2. Đề xuất downstream-aware objective để đánh giá prompt.
3. Đề xuất training-free subset selection có redundancy control.
4. Kết hợp Grounding DINO regions với GeoCLIP attention intervention.
5. Đánh giá đồng thời geolocation performance và explanation faithfulness.
6. Phân tích các prompt/object được chọn để hiểu loại visual clues GeoCLIP sử dụng.

---

## 19. Pipeline khuyến nghị hiện tại

```text
Candidate prompt/object pool
        ↓
Grounding DINO text embeddings + ID
        ↓
Single-candidate screening
        ↓
Remove weak and redundant candidates
        ↓
Greedy forward subset selection
        ↓
Backward deletion + one-swap refinement
        ↓
Fixed vocabulary subset
        ↓
Search GeoCLIP intervention parameters
        ↓
Held-out evaluation
        ↓
Analyze selected prompts, regions and causal effects
```

## 20. Ý tưởng quan trọng nhất

Vocabulary không nên được chọn chỉ vì Grounding DINO phát hiện object đó tốt.

Vocabulary nên được chọn vì nó tạo ra các region:

- ổn định;
- vừa đủ cụ thể;
- không dư thừa;
- không bao phủ quá nhiều ảnh;
- có hiệu ứng intervention đáng kể;
- và giúp giải thích hoặc cải thiện đầu ra GeoCLIP.

Nói cách khác, objective chính là **downstream causal usefulness**, không phải detection confidence đơn thuần.
