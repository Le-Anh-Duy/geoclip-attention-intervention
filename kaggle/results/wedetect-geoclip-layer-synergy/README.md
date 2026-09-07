# Tổng hợp thí nghiệm WeDetect–GeoCLIP layer search và layer synergy

## 1. Mục tiêu

Thí nghiệm kiểm tra hai câu hỏi:

1. Layer nào của image encoder GeoCLIP phản ứng có lợi nhất khi attention được
   điều hướng vào các object proposal của WeDetect?
2. Nếu nhiều layer riêng lẻ có kết quả tốt, intervention đồng thời các layer đó
   có tạo hiệu ứng cộng hưởng tốt hơn layer đơn hay không?

Kết quả trong tài liệu này đến từ:

- `giahuytran1/wedetect-img2gps3k-cache`, version 2;
- `giahuytran1/wedetect-geoclip-signal-search`, version 1;
- `giahuytran1/wedetect-geoclip-layer-synergy`, version 2.

Các notebook chạy offline trên NVIDIA RTX PRO 6000 Blackwell Server Edition.
Version 3 deterministic FP32 của notebook synergy đã được push để kiểm tra lại
độ tái lập, nhưng kết quả của version đó chưa được dùng trong báo cáo này.

## 2. Dữ liệu và intervention

Tập đánh giá là toàn bộ 2.997 ảnh Img2GPS3K. WeDetect chỉ được chạy một lần rồi
cache proposal để tất cả thí nghiệm GeoCLIP dùng cùng một đầu vào detector.

- WeDetect cache threshold: 0,15;
- NMS IoU: 0,7;
- tối đa 100 proposal mỗi ảnh;
- thời gian chạy detector: 19,8 giây.

Trong search mặc định, proposal có score từ 0,4 trở lên được hợp thành một mask
duy nhất. Mask được chiếu lên lưới 16×16 patch của CLIP ViT-L/14 sau resize và
center crop.

Intervention được cộng vào attention logits trước softmax:

```text
key patch nằm trong proposal mask: a = +2
key patch nằm ngoài proposal mask: b = -2
CLS key:                            0
```

Mỗi layer candidate được intervention trên toàn bộ 16 attention head. Delta theo
từng ảnh được định nghĩa là:

```text
delta = baseline geolocation error - intervention geolocation error
```

Delta dương nghĩa là intervention giảm lỗi. Tiêu chí search là trung bình delta
sau khi clip từng ảnh về khoảng [-2.500, +2.500] km, nhằm giảm ảnh hưởng của một
vài bước nhảy xuyên lục địa.

## 3. Discovery và locked holdout

Ảnh được chia cố định theo SHA-256 của tên file:

```text
SHA256(filename) mod 3 == 0  -> discovery
còn lại                      -> holdout
```

| Split | Số ảnh | Vai trò |
|---|---:|---|
| Discovery | 1.079 | Tìm và chọn cấu hình |
| Locked holdout | 1.918 | Chỉ đánh giá cấu hình đã chọn |

Discovery score là clipped mean delta trên 1.079 ảnh discovery. Holdout score là
cùng metric trên 1.918 ảnh chưa dùng để chọn cấu hình. Một candidate tốt nhất khi
nhìn trực tiếp vào holdout chỉ là kết quả hậu nghiệm, không phải kết quả xác nhận.

## 4. Search từng layer

Search đầu tiên intervention lần lượt từng layer 0–23, mỗi lần tác động cả 16
head. Discovery chọn layer 2. Kết quả locked holdout của cấu hình này trong lần
search đầu tiên là:

| Metric | Baseline | Layer 2 |
|---|---:|---:|
| Mean error (km) | 1.984,157 | 1.976,694 |
| Median error (km) | 390,130 | 395,441 |
| Acc@25 | 0,09854 | 0,09750 |
| Acc@200 | 0,36757 | 0,36496 |
| Acc@750 | 0,62357 | 0,62252 |
| Acc@2500 | 0,80396 | 0,80448 |

- Raw mean improvement: +7,464 km;
- clipped mean improvement: +2,602 km;
- bootstrap 95% CI raw: [-29,86; +42,92] km;
- bootstrap 95% CI clipped: [-10,93; +16,75] km.

Khoảng tin cậy cắt qua 0, do đó chưa có bằng chứng rằng layer 2 cải thiện accuracy.
Kết quả rõ hơn là độ nhạy tăng theo chiều sâu: tương quan giữa layer index và
holdout clipped delta là r=-0,818. Các late layer thường bị phá mạnh hơn khi nhận
cùng một intervention.

## 5. Thiết kế search layer synergy

Từ search trước, năm layer có discovery score dương là `2, 4, 7, 12, 3`, theo
thứ tự score giảm dần. Với năm layer này, thí nghiệm synergy sinh tất cả tập con
từ 2 đến 5 layer:

```text
C(5,2) + C(5,3) + C(5,4) + C(5,5) = 26 tổ hợp layer
```

Mỗi tổ hợp được thử với ba cách scale bias:

| Scale | Magnitude tại mỗi layer | Mục đích |
|---|---:|---|
| Fixed | `2` | Kiểm tra cộng hưởng khi giữ nguyên intervention mỗi layer |
| Sqrt | `2/sqrt(k)` | Gần giữ cố định năng lượng perturbation |
| Linear | `2/k` | Gần giữ cố định tổng liều intervention |

Với `k` là số layer. Tổng cộng có 78 tổ hợp và 5 single-layer control chạy lại
trong cùng process. Tổ hợp chỉ được chọn bằng discovery.

## 6. Kết quả xác nhận synergy

Tổ hợp thắng discovery là:

```text
layers:       2 + 4 + 3
heads:        toàn bộ 16 head
scale:        fixed
bias/layer:   +2 / -2
discovery:    +22,962 km clipped mean delta
```

Khi mở locked holdout:

| Cấu hình | Raw mean delta (km) | Clipped mean delta (km) | Mean prediction shift (km) |
|---|---:|---:|---:|
| Layer 2 đơn, cùng runtime | +15,108 | +3,830 | 173,970 |
| Layers 2+4+3 | +22,872 | +3,776 | 350,373 |
| Synergy trừ layer 2 | — | **-0,054** | — |

Kiểm định bootstrap 10.000 lần:

| Đại lượng | 95% CI (km) |
|---|---:|
| Raw delta của 2+4+3 | [-35,61; +81,62] |
| Clipped delta của 2+4+3 | [-18,87; +26,15] |
| 2+4+3 trừ layer 2, paired clipped | [-20,28; +19,70] |

Tổ hợp thắng discovery không tốt hơn layer 2 trên holdout. Chênh lệch -0,054 km
gần bằng 0 và khoảng tin cậy paired rất rộng. Vì vậy giả thuyết cộng hưởng chưa
được xác nhận.

Các accuracy của cấu hình locked 2+4+3 là:

| Metric | Giá trị |
|---|---:|
| Acc@1 | 0,00209 |
| Acc@25 | 0,10375 |
| Acc@200 | 0,37122 |
| Acc@750 | 0,62200 |
| Acc@2500 | 0,80136 |

## 7. Xu hướng theo số layer và scale

Clipped score trung bình của toàn bộ candidate trên holdout:

| Scale | 2 layer | 3 layer | 4 layer | 5 layer |
|---|---:|---:|---:|---:|
| Fixed | -16,93 | -25,71 | -27,97 | -30,36 |
| Sqrt | -11,93 | -9,29 | -14,62 | -12,11 |
| Linear | -6,87 | -3,03 | -0,78 | +5,71 |

Fixed bias xấu dần khi thêm layer, cho thấy perturbation tích lũy thường lấn át
lợi ích của từng layer. Linear scaling an toàn hơn, nhưng phần lớn chỉ hạn chế
thiệt hại; nó chưa chứng minh hiệu ứng cộng hưởng. Tương quan ranking candidate
giữa discovery và holdout chỉ r=0,140, nghĩa là lựa chọn tổ hợp rất không ổn định.

## 8. Kết quả hậu nghiệm đáng chú ý

Các cấu hình dưới đây được xếp hạng bằng chính holdout nên chỉ dùng để sinh giả
thuyết cho lần chạy xác nhận tiếp theo:

| Cấu hình | Holdout clipped delta (km) | Gain so với constituent tốt nhất (km) |
|---|---:|---:|
| Fixed layers 2+3 | +11,565 | +2,431 |
| Linear layers 4+7 | +10,141 | +9,119 |
| Fixed layers 2+7 | +6,454 | +2,623 |
| Sqrt layers 2+7 | +5,851 | +2,021 |
| Linear layers 2+4+7+12+3 | +5,708 | -3,426 |

Ngay cả hai candidate đầu cũng chưa có bằng chứng thống kê:

- `2+3`: clipped CI [-6,18; +29,43] km; paired gain CI
  [-13,28; +19,14] km;
- linear `4+7`: clipped CI [-7,33; +27,02] km; paired gain CI
  [-7,58; +24,89] km.

Không được báo cáo `2+3` hoặc `4+7` như cấu hình thắng, vì chúng được nhận ra sau
khi đã xem holdout.

## 9. Cảnh báo về độ tái lập số học

Version 2 dùng BF16/TF32 để tận dụng RTX PRO 6000. Khi baseline được tính lại ở
process mới, 1.507/2.997 ảnh có error lệch trên 0,1 km so với lần trước; mean
absolute difference là 44,81 km và trường hợp lớn nhất là 18.413,21 km. Nguyên
nhân có thể là nhiều gallery embedding có similarity rất sát nhau: sai số số học
nhỏ đủ đổi nearest neighbour, từ đó tạo bước nhảy GPS rất lớn.

Các so sánh synergy của version 2 vẫn là paired trong cùng runtime, vì baseline,
single controls và combinations được chạy cùng process. Tuy nhiên, kết quả chưa
nên được coi là kết luận cuối cùng trước khi version deterministic FP32 hoàn tất.

## 10. Kết luận hiện tại

1. Chưa có bằng chứng nhiều layer tốt tạo cộng hưởng tổng quát hóa.
2. Tổ hợp được chọn trên discovery không thắng layer đơn trên locked holdout.
3. Thêm nhiều layer với fixed bias thường làm kết quả xấu hơn do perturbation
   tích lũy.
4. `2/k` là cách scale an toàn nhất trong các thử nghiệm này, nhưng chưa thể hiện
   synergy rõ ràng.
5. Cần dùng kết quả deterministic FP32 và một split xác nhận mới trước khi đưa ra
   claim về accuracy hoặc một tổ hợp layer cụ thể.

Artifact đầy đủ của version 2 nằm trong thư mục ignored:
`outputs/wedetect-geoclip-layer-synergy-v2/`.
