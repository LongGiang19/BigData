# Kiểm toán đánh giá mô hình

Nguồn là model, cấu hình, curated Validation và predictions của lượt kiểm chứng `full_20261109_215910_140315`. Test có 1.162.581 dòng ở 7 ngày, gồm 902.988 Benign và 259.593 Attack. Tỷ lệ Attack gộp là 22,329%. Các phép tính dùng điều kiện cảnh báo hiện tại `score > threshold`.

| Mô hình | TP | FP | TN | FN | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 2.215 | 26.959 | 876.029 | 257.378 | 7,5924% | 0,8533% | 1,5341% | 2,9855% | 0,657854 | 0,268816 |
| Autoencoder | 616 | 246 | 902.742 | 258.977 | 71,4617% | 0,2373% | 0,4730% | 0,0272% | 0,796963 | 0,413150 |
| Ensemble | 872 | 4.214 | 898.774 | 258.721 | 17,1451% | 0,3359% | 0,6589% | 0,4667% | 0,739863 | 0,334288 |

Audit tính PR-AUC bằng tích phân hình thang trên toàn bộ điểm gãy của đường PR, đồng thời lưu Average Precision riêng. Recall và FPR khớp metrics Spark đã lưu. PR-AUC có chênh lệch nhỏ do cách dựng đường cong, đáng kể nhất ở AE với 0,413150 trong audit và 0,412205 trong metrics Spark.

## Nhãn, điểm và hiệu chuẩn

- Kiểm tra toàn bộ predictions cho thấy 0 sai khác giữa `y` và quy tắc chuẩn hóa `label_raw`: Benign = 0, nhãn khác = 1. Quy tắc cũng khớp trên toàn bộ 591.873 dòng Validation. Đây là kiểm tra tính nhất quán với nhãn nguồn.
- IF đảo dấu `score_samples`, vì vậy điểm cao tương ứng với mẫu bất thường hơn. AE dùng trung bình bình phương sai số tái tạo trên các features. Ensemble lấy trung bình hai log-score đã chuẩn hóa, với hai độ lệch chuẩn dương. Tái tính Ensemble trên tất cả predictions cho sai số tuyệt đối tối đa 5,33×10⁻¹⁵. Toàn bộ cờ Ensemble khớp điều kiện so ngưỡng.
- Đọc lại luồng pipeline và tính điểm trên đúng 446.619 dòng Benign của 16/02. Phân vị 99% tái tạo ngưỡng IF 0,583428691 và AE 2,077148290 chính xác. Ngưỡng Ensemble tái tính 5,185228145 so với 5,185227849 đã lưu, chênh 2,96×10⁻⁷ khi tái chấm điểm theo batch. FPR trên Benign Validation tại ngưỡng đã lưu là 0,9995%, 1,0002% và 1,0002% theo thứ tự. Kết quả xác nhận quy trình dùng Benign Validation và đạt mục tiêu FPR tại ngày hiệu chuẩn.

## Vì sao Recall thấp

- **Điểm vận hành quá cao cho nhiều Attack trên test.** AE có ROC-AUC 0,7970 và PR-AUC 0,4131 so với prevalence 0,2233, nhưng chỉ 616/259.593 Attack vượt ngưỡng. Ngày 20/02 là bằng chứng rõ nhất: PR-AUC AE 0,9388 so với prevalence 0,6015, trong khi Recall chỉ 0,1339%. Ngày 02/03, AE đạt PR-AUC 0,6443 so với prevalence 0,1884 nhưng không phát hiện dòng Bot nào tại ngưỡng hiện tại.
- **Giới hạn xếp hạng thay đổi theo ngày và loại attack.** Ngày 21/02, PR-AUC cả ba chỉ 0,2060–0,2119, dưới prevalence 0,3552. HOIC có 49.499 dòng và cả ba bắt 0. Ngược lại, AE bắt 424/424 LOIC-UDP và Ensemble bắt 421/424. LOIC-HTTP và HOIC chiếm 192.870/259.593 Attack (74,30%), nên kết quả thấp ở hai nhóm này chi phối Recall gộp.
- **Trade-off của Ensemble.** Ensemble có ROC-AUC và PR-AUC gộp thấp hơn AE. IF bắt 1.897/29.863 Infilteration, còn Ensemble bắt 193 và AE bắt 0. Kết quả cho thấy việc gộp điểm hiện tại không giữ được lợi thế phát hiện của từng mô hình trên mọi nhóm. Tập predictions không đủ để tách ảnh hưởng trọng số khỏi ảnh hưởng ngưỡng.
- **FPR thực tế biến động theo ngày.** IF vượt 5,34% ngày 28/02 và 01/03, trong khi AE không vượt 0,1843% ở bất kỳ ngày test nào. Ensemble vượt 1% ngày 01/03. Điều này cho thấy điểm vận hành hiệu chuẩn vào 16/02 không chuyển nguyên vẹn sang test. Calibration thực thi đúng, nhưng mức FPR mục tiêu chỉ được xác nhận tại ngày hiệu chuẩn.

Các bảng [theo ngày](by_day.csv) và [theo loại tấn công](by_attack.csv) chứa đủ 21 nhóm ngày–mô hình và 24 nhóm attack–mô hình, cùng số lượng để tránh diễn giải quá mức các nhóm nhỏ.

## Vì sao số dòng test khác nhau

Artifacts trước sửa có 1.164.788 predictions, lượt kiểm chứng có 1.162.581, chênh 2.207. Hai lượt dùng cùng `TEST_FRAC=0.25`, seed cấu hình 42 và cùng ngày test. `sample(False, frac, seed)` là lấy mẫu Bernoulli, với fraction là tỷ lệ kỳ vọng. Spark dùng luồng ngẫu nhiên theo partition, bao gồm `seed + partitionIndex` ở nhánh codegen, nên cùng seed không đảm bảo cùng tập dòng khi phân vùng hoặc thứ tự dòng khác. Các lượt đã ghi lại các file curated với ID file khác nhau. Artifact hiện có thiếu physical plan và định danh dòng nguồn để xác định partition hoặc dòng cụ thể đã đổi. [Spark SampleExec](https://raw.githubusercontent.com/apache/spark/v3.5.8/sql/core/src/main/scala/org/apache/spark/sql/execution/basicPhysicalOperators.scala#L288)

Để khóa test sample, chốt manifest phiên bản dữ liệu nguồn, cấp ID bền vững trước repartition, chẳng hạn mã file nguồn bất biến cộng row-group và ordinal trong file. Tạo sample bằng băm ID với seed, hoặc chọn đúng quota mỗi ngày bằng thứ tự băm có tie-breaker là ID nếu cần số lượng chính xác. Lưu ID và Parquet sample một lần, kèm hash manifest, số dòng và phân bố nhãn để mọi lượt dùng lại cùng sample.

## Kết luận và kiểm chứng tiếp

- Audit không phát hiện lỗi mapping, đảo chiều điểm, tính confusion matrix hoặc dùng sai tập hiệu chuẩn trong những đường thực thi đã kiểm tra.
- Giữ một test sample bất biến cho so sánh mô hình. Nếu cần nghiên cứu lại ngưỡng, đo độ ổn định trên tập calibration theo thời gian độc lập với test hiện tại. Phân tích riêng các nhóm HOIC, LOIC-HTTP và Bot trước khi xem xét thay đổi mô hình.
- Chưa xác minh nguyên nhân dữ liệu khiến các phân phối điểm thay đổi, chất lượng nhãn gốc, hay quan hệ nhân quả của từng feature. Predictions thiếu ID dòng để tái ghép từng test flow với input và chấm lại IF/AE trên đúng các dòng đó. Kiểm tra hướng IF/AE dựa trên mã scoring và lần tái chấm toàn bộ Benign Validation; kiểm tra Ensemble và cờ cảnh báo bao phủ toàn bộ test.
- Các biểu đồ PR và ROC đánh dấu điểm vận hành đang dùng. Đường cong mô tả xếp hạng, không được dùng để chọn ngưỡng mới.
