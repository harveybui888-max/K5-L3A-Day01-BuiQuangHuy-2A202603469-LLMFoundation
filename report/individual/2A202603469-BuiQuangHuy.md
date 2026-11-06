# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☑   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | ☑   |

## 2. Kết quả chính

Mô hình F1 test F1 SUB 40 p95 (ms/tin) USD/1k tin

1. BoW + LogReg Chưa có 0.857 Chưa có 0
2. BoW + MLP 0.945 0.974 0.43\* 0
   3a. Emb + kNN Chưa có 0.923 Chưa có 0
   3b. Emb + LogReg Chưa có 0.919 Chưa có 0
   3c. Emb + MLP head Chưa có 0.919 Chưa có 0
   4a. LLM Zero-shot Chưa có Chưa có Chưa có Chưa có
   4b. LLM Few-shot Chưa có Chưa có Chưa có Chưa có

## 3. Dự đoán và thực tế

- Dự đoán: BoW + LogReg đạt accuracy 60–70%, BoW + MLP 62–72%, Emb + LogReg 72–82%.
- Thực tế: Lần lượt đạt 52%, 56% và 68%.
- Nguyên nhân: Bộ hard cases chứa nhiều tin cố tình đánh lừa mô hình, chẳng hạn spam không có từ khóa và ham chứa từ khóa spam. Tôi đã đánh giá quá cao khả năng tổng quát hóa, đặc biệt của BoW.
- Dự đoán: BoW + MLP có p95 lớn hơn 50 ms, trong khi Embedding có thể dưới 50 ms.
- Thực tế: BoW + MLP có p95 khoảng 0.43 ms trong phép đo trước đó, thấp hơn SLA 50 ms rất nhiều.
- Nguyên nhân: Tôi chưa đánh giá đúng tốc độ suy luận của MLP nhỏ. Đồng thời, tôi chưa phân biệt rõ thời gian chạy classifier với thời gian của toàn bộ pipeline tạo embedding.
- Dự đoán: MLP có thể đạt F1 tương đương hoặc thấp hơn LogReg vì có nhiều tham số và dữ liệu huấn luyện ít.
- Thực tế: MLP đạt F1 test 0.945, còn F1 SUB là 0.974, cao hơn BoW + LogReg trên SUB (0.857). Validation loss thấp nhất ở epoch 5.
- Nguyên nhân: Tôi đánh giá thấp khả năng học quan hệ phi tuyến của MLP. Tuy nhiên, validation loss tăng sau giai đoạn đầu cũng cho thấy nguy cơ overfitting, nên cần early stopping.

## 4. Phân tích 50 case khó

Mô hình
F1 SUB thường
F1 hard cases
Mức giảm
BoW + LogReg
0.857
0.333
52.4 điểm %
BoW + MLP
0.974
0.450
52.4 điểm %
Emb + kNN
0.923
0.737
18.6 điểm %
Emb + LogReg
0.919
0.692
22.7 điểm %
Emb + MLP head
0.919
0.694
22.5 điểm %
BoW + LogReg và BoW + MLP cùng giảm nhiều nhất, 52.4 điểm phần trăm F1.

Những nhóm có khả năng gây khó khăn lớn cho BoW là ham chứa từ khóa spam, spam giọng tự nhiên và né từ khóa. Chưa có bảng kết quả theo từng nhóm nên chưa thể xác định nhóm có tỷ lệ sai cao nhất.

Hai ví dụ để phân tích cơ chế lỗi:

Case 1 — Ham chứa từ khóa spam
I WON the FREE pizza challenge yesterday! Claim your slice when you come over tonight.
Nhãn đúng: ham.

BoW có nguy cơ dự đoán spam vì xuất hiện WON, FREE, Claim. Trong khi đó, nội dung thực chất là lời nhắn bạn bè về một cuộc thi ăn pizza. Mô hình dựa trên ngữ nghĩa có thể hiểu ngữ cảnh tốt hơn.

Case 2 — Spam giả dạng người quen
Hey, I changed my number. I'm stuck outside and need you to transfer some money to this account.
Nhãn đúng: spam trong bộ thử nghiệm giả định.

BoW có nguy cơ dự đoán ham vì văn bản giống cuộc trò chuyện thông thường, thiếu từ khóa quảng cáo. Embedding có thể nhận diện ý định yêu cầu chuyển tiền đáng ngờ.

Hai ví dụ này minh họa cơ chế gây lỗi; chưa có kết quả dự đoán từng case để xác nhận mô hình thực sự phân loại sai chúng

## 5. Thử thách cá nhân

tạo 4 tin nhắn:
Tin nhắn Nhãn đúng Điểm gây khó
"Hi, your account needs a quick identity check. Please enter your login details." Spam Lừa lấy thông tin đăng nhập, không dùng từ khóa quảng cáo
"Hey, I changed my number. Can you transfer some money? I'll pay you back tomorrow." Spam Giả dạng người quen
"Are you free this evening? We can grab dinner after work." Ham free nghĩa là rảnh
"Did you win the chess match yesterday? Your team played really well!" Ham win nghĩa là chiến thắng trong trò chơi

Tôi sử dụng vectorize() và predict_proba() để phân loại bằng BoW + LogReg; với Embedding + LogReg, tôi tạo vector bằng embed(), chuẩn hóa bằng mu, sd, rồi dự đoán với w_e, b_e.

Chưa có output dự đoán riêng cho 4 tin nhắn này nên chưa thể xác nhận model nào đúng từng case.

Tôi kỳ vọng Embedding + LogReg xử lý tốt hơn vì biểu diễn toàn bộ ý nghĩa của câu, trong khi BoW dựa trên các từ riêng lẻ. Tuy nhiên, Embedding cũng có thể nhầm lẫn nếu một tin nhắn giả dạng người quen có ngữ nghĩa gần với các tin ham trong tập huấn luyện.

## 6. Phương án cho sản phẩm thật

Yêu cầu: Lọc 10 triệu SMS/ngày, p95 dưới 50 ms/tin.
Lưu lượng trung bình
\[ \frac{10{,}000{,}000}{24\times60\times60}\approx115.74 \]
Tôi đề xuất BoW + MLP làm bộ lọc chính, kết hợp Embedding + LogReg cho những trường hợp không chắc chắn.

Lý do:

- BoW + MLP đạt F1 test 0.945 và p95 khoảng 0.43 ms/tin trong phép đo trước đó, thấp hơn SLA 50 ms.
- Embedding + LogReg có F1 trên hard cases là 0.692, cao hơn BoW + MLP (0.450).
- Chỉ đưa những tin có xác suất dự đoán gần 0.5 sang Embedding để giảm tải xử lý.

Phương án này tăng độ phức tạp triển khai và có thể làm tăng độ trễ ở những tin phải qua Embedding. Cần đo p95 end-to-end của router trên tải thực tế, bao gồm bước tạo embedding, trước khi khẳng định toàn hệ thống đạt SLA.

Tôi không chọn LLM API làm bộ lọc chính vì chưa có số liệu chứng minh p95 dưới 50 ms và chi phí API sẽ tăng theo lưu lượng. LLM phù hợp hơn để hỗ trợ phân tích các trường hợp khó theo luồng xử lý riêng, không nằm trên đường xử lý đồng bộ bắt buộc.

## 7. AI log và điều đã học

. LLM Viết API, prompt, parse label và batch classification Hoàn thiện TODO 4.1–4.4 check format
Hiểu rrằng chọn mô hình cho sản phẩm thật không chỉ dựa vào F1, mà còn phải xét độ trễ p95, chi phí, khả năng tổng quát hóa và yêu cầu vận hành. Một hệ thống kết hợp nhiều mô hình có thể hợp lý hơn việc luôn sử dụng mô hình lớn nhất.
