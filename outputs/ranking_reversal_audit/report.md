# Kiểm chứng ranking reversal ngày 21/02/2018

Audit đọc duy nhất partition `day=2018-02-21` của predictions full test chính thức
`test100_20261109_233550_850520`. Có **561.396 flow**, gồm **360.805 Benign**
và **200.591 Attack**. Tỷ lệ Attack là **35,7307%**. Nhãn nhị phân khớp toàn bộ
với `label_raw` sau chuẩn hóa: Benign = 0, Attack = 1.

Pipeline định nghĩa hướng bất thường cao hơn bằng `-score_samples` cho IF,
reconstruction MSE cho AE và trung bình hai log-score chuẩn hóa cho Ensemble.
Hai độ lệch chuẩn Ensemble dương. Tái tính Ensemble từ hai cột score và
thống kê đã lưu khớp với sai số tối đa **3,55×10⁻¹⁵**. Toàn bộ cờ
Ensemble khớp `score > threshold`.

| Mô hình | ROC-AUC gốc | ROC-AUC với −score | PR-AUC gốc | PR-AUC với −score |
|---|---:|---:|---:|---:|
| Isolation Forest | 0,016723 | 0,983277 | 0,207596 | 0,974981 |
| Autoencoder | 0,008830 | 0,991170 | 0,213444 | 0,991728 |
| Ensemble | 0,008861 | 0,991139 | 0,213434 | 0,993028 |

ROC-AUC dùng scikit-learn với Attack là lớp dương. PR-AUC là diện tích
hình thang dưới đường Precision–Recall. PR-AUC gốc của cả ba dưới đường
nền tỷ lệ Attack 0,357307. Tổng AUC gốc và AUC đảo dấu bằng 1 cho cả ba,
kể cả khi tính đồng hạng bằng nửa cặp. Đảo dấu chỉ là đối chiếu hậu nghiệm.

| Mô hình | Median Benign | Median Attack | p99 Benign | p99 Attack |
|---|---:|---:|---:|---:|
| Isolation Forest | 0,548302 | 0,510733 | 0,554101 | 0,529425 |
| Autoencoder | 0,076409 | 0,010435 | 0,082995 | 0,024537 |
| Ensemble | 0,691730 | −2,847675 | 0,980325 | −1,381216 |

Đồ thị CDF dùng tất cả điểm. Median Attack thấp hơn Benign ở cả ba
mô hình, nhưng đuôi Attack vẫn có một nhóm score cao.

## Phân rã theo loại Attack

Mỗi AUC theo loại dùng cùng 360.805 Benign. AUC gộp bằng trung bình
AUC theo loại, trọng số theo tỷ lệ Attack từng loại. Đóng góp vào phần
xếp sai có trọng số `tỷ lệ loại × (1 − AUC loại)`, gồm nửa số cặp hòa.

| Loại | Số flow | Tỷ lệ Attack | AUC IF | AUC AE | AUC Ensemble |
|---|---:|---:|---:|---:|---:|
| DDOS attack-HOIC | 198.861 | 99,1375% | 0,008216 | 0,000207 | 0,000239 |
| DDOS attack-LOIC-UDP | 1.730 | 0,8625% | 0,994594 | 1,000000 | 0,999993 |

HOIC chiếm **99,9953%** phần xếp sai có trọng số của IF, **100%**
của AE và khoảng **99,999994%** của Ensemble. Không dòng HOIC nào vượt
ngưỡng đã lưu. Với LOIC-UDP, số vượt ngưỡng lần lượt là 25, 1.730 và
1.721, dù khả năng xếp hạng gần hoàn hảo. Điều này tách được hiệu ứng
xếp hạng của từng loại khỏi hiệu ứng ngưỡng.

## Kết luận dùng trong báo cáo khoa học

> Trên toàn bộ 561.396 flow của ngày 21/02/2018, ba điểm bất thường đều
> thể hiện đảo chiều xếp hạng so với nhãn dataset. ROC-AUC nằm trong
> khoảng 0,00883–0,01672. Phân rã theo loại cho thấy hiện tượng gộp chủ
> yếu đến từ HOIC, nhóm chiếm 99,14% flow Attack và có AUC dưới 0,009
> ở cả ba mô hình. LOIC-UDP cùng ngày vẫn được xếp hạng tốt, với AUC
> ít nhất 0,99459. Kết quả xác nhận đảo chiều trong cohort này dưới
> định nghĩa score hiện tại, với phạm vi giới hạn ở ngày và nhãn đã quan sát.

Các giả thuyết gồm feature của HOIC rơi vào vùng điểm thấp của model,
phân phối Benign của ngày này tạo điểm cao hơn tương đối, hoặc hiệu ứng
từ cách tạo feature và gán nhãn dataset. Audit chỉ dùng score đã lưu nên
chưa xác định cơ chế nào. Cần đối chiếu feature theo flow, kiểm chứng nhãn
và có cửa sổ thời gian độc lập để đánh giá giả thuyết. AUC đảo dấu cao
không chứng minh đảo dấu sẽ hữu ích ở ngày khác hay là lựa chọn triển khai.
