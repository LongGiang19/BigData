# FPR Sensitivity Study

Nguồn là model và predictions của lượt full test `test100_20261109_233550_850520`.
Tái chấm **446.619 Benign Validation** ngày 16/02 bằng model và thông số
chuẩn hóa Ensemble đã lưu. Mỗi ngưỡng là phân vị `1 − FPR target` trên
Validation. Cảnh báo khi `score > threshold`. Mốc 1% khớp threshold đã lưu
trong dung sai 2×10⁻⁶ và tái tạo chính xác confusion matrix của cả ba mô hình.

Áp dụng 15 tổ hợp mô hình–ngưỡng lên cùng **4.653.501 flows**, gồm 1.039.855
Attack và 3.613.646 Benign. Tổng TP/FP/TN/FN theo ngày khớp bảng gộp. Tổng
TP/FN theo attack cũng khớp. Trong bảng theo attack, từng loại được đối
chiếu với cùng toàn bộ pool Benign test. FP, TN và FPR được lặp lại cho
mỗi loại. Precision và F1 theo loại phụ thuộc pool tham chiếu này, và
không được cộng gộp giữa các loại.

## Recall và FPR test

Mỗi ô là **Recall / FPR test** theo phần trăm.

| FPR target trên Validation | Isolation Forest | Autoencoder | Ensemble |
|---:|---:|---:|---:|
| 0,1% | 0,6763 / 2,2019 | 0,2421 / 0,0177 | 0,2782 / 0,2127 |
| 0,5% | 0,8254 / 2,8227 | 0,2421 / 0,0204 | 0,2946 / 0,2501 |
| 1% | 0,8623 / 2,9960 | 0,2421 / 0,0245 | 0,3402 / 0,4657 |
| 2% | 1,3399 / 4,2858 | 0,2426 / 0,0342 | 1,0352 / 1,8852 |
| 5% | 2,5256 / 17,3232 | 0,5884 / 0,6701 | 1,3890 / 3,1719 |

- **IF:** Mục tiêu 5% cho Recall cao nhất trong dải khảo sát, với threshold
  khoảng 0,534246. So với mốc 1%, phát hiện thêm 17.296 Attack và thêm
  517.735 FP. Precision giảm từ 7,65% xuống 4,03%. IF vượt FPR mục tiêu
  trên test ở cả năm mốc. Ngày 21/02 tại mục tiêu 5%, FPR lên 77,57%
  trong khi Recall chỉ 0,86%.
- **AE:** Các mốc 0,1%, 0,5% và 1% đều bắt 2.517 Attack. FP lần lượt
  là 640, 736 và 887. Mốc 5% với threshold khoảng 0,197527 bắt 6.118
  Attack, Recall 0,5884%, FPR 0,6701% và Precision 20,17%. FPR test
  thấp hơn mục tiêu tại cả năm mốc, đồng thời Recall vẫn thấp.
- **Ensemble:** Từ mục tiêu 1% lên 2%, Recall tăng từ 0,3402% lên 1,0352%,
  FPR từ 0,4657% lên 1,8852%. Mốc 5% với threshold khoảng 1,026777 có
  Recall 1,3890%, FPR 3,1719% và Precision 11,19%. Ensemble vượt mục
  tiêu ở mốc 0,1% và thấp hơn mục tiêu ở bốn mốc còn lại.

## Xếp hạng theo ngày và khả năng chuyển ngưỡng

ROC-AUC và PR-AUC theo ngày được tính từ score, độc lập với ngưỡng. Ở
mốc 1%, AE ngày 20/02 có ROC-AUC **0,9612**, PR-AUC **0,9369** so với
tỷ lệ Attack **60,26%**, nhưng Recall chỉ **0,1368%**. Ngày 02/03 có
ROC-AUC **0,9515** và PR-AUC **0,6445** so với tỷ lệ Attack **18,73%**,
nhưng Recall bằng **0**. Ensemble ngày 02/03 có ROC-AUC **0,9293**,
PR-AUC **0,5598** và Recall **0,0803%**. Điểm vận hành nằm ngoài vùng
phát hiện phần lớn Attack ở các ngày này.

FPR trên Benign Validation bám sát cả năm mục tiêu. FPR trên Test thay
đổi đáng kể theo ngày. Ở mục tiêu 1%, dải FPR theo ngày là **0,5355%–5,4365%**
cho IF, **0%–0,1571%** cho AE và **0,0826%–1,2611%** cho Ensemble.
Chênh lệch này cho thấy ngưỡng chuyển từ ngày 16/02 sang các ngày Test
không duy trì cùng tỷ lệ cảnh báo Benign. Audit chưa xác định nguyên
nhân làm phân phối score thay đổi.

Ở mục tiêu 5%, HOIC vẫn chỉ có **1/198.861** TP với IF và **0** với AE
và Ensemble. LOIC-HTTP có Recall khoảng **0,14%** ở cả ba mô hình.
Điều này cho thấy nới ngưỡng trong dải khảo sát có giới hạn rõ đối
với các nhóm Attack lớn, dù Recall gộp tăng.

Kết quả chỉ mô tả năm ngưỡng đã định trước trên một snapshot và một
ngày calibration. Nếu cần chọn điểm vận hành, hãy xác định ngân sách FP
và đánh giá trên các ngày calibration độc lập trước khi khóa ngưỡng.
Kết quả Test trong báo cáo không dùng để chọn ngưỡng triển khai.

## Tái lập

Chạy `study.py` bằng Python của dự án. Dùng `--output` trỏ tới thư mục
mới khi chạy lại để giữ artifacts đã có. Các CSV chứa 15 hàng gộp,
105 hàng theo ngày, 120 hàng theo loại attack và 21 hàng ranking theo ngày.
