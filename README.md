# GeoCLIP attention experiments

Kho thực nghiệm cho đề tài **can thiệp attention dựa trên vùng phát hiện để
định vị địa lý bằng ảnh có thể giải thích** của Lê Anh Duy và Trần Gia Huy.

Mỗi thư mục cấp cao nhất là một thử nghiệm hoặc một nguồn phụ thuộc độc lập:

- `geo-clip/`: mã nguồn GeoCLIP dùng làm baseline cố định;
- `test-intervention/`: sandbox can thiệp attention trước softmax bằng vùng
  chọn thủ công, tiền thân của vùng phát hiện từ Grounding DINO;
- các thử nghiệm mới: tạo thêm một thư mục mới ngay tại cấp này.

## Phạm vi đánh giá

- GeoCLIP baseline và phương pháp đề xuất dùng cùng location encoder và gallery;
- độ chính xác theo Haversine/GCD và Acc@1/25/200/750/2500 km;
- độ bền trước thay đổi miền, nhiễu có kiểm soát và lỗi detector;
- tính trung thực nhân quả qua mask/keep/random-region controls;
- thời gian chạy, thông lượng và bộ nhớ trên cùng phần cứng.

## Quy ước cho thử nghiệm mới

Mỗi thư mục thử nghiệm tự chứa code/config cần thiết và một `README.md` ghi tối
thiểu: giả thuyết, dữ liệu/phiên bản, seed, phần cứng, lệnh chạy và kết quả.
Dữ liệu thô, môi trường ảo, cache, log, checkpoint và output lớn không được đưa
lên Git; chỉ commit cấu hình, mã nguồn và kết quả tóm tắt đủ để tái lập.
