# XGBoost_explain_and_example
# Lý thuyết gradient tree boosting
Về cơ bản, XGBoost sử dụng phương pháp boosting để kết hợp các cây, nên giá trị đầu ra có thể viết như sau:
 ![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/0f84242a-e822-4cdf-acf6-bc710555b1c2)

Trong đó fk biểu thị cho mô hình cơ bản thứ k và y_hat là giá trị dự đoán của mẫu thứ i .
Nói một cách đơn giản, chúng ta cần học từng cây, và về cơ bản là học như thế nào? đương nhiên là vẫn giống như các phương pháp học máy khác đó là  xuất phát từ hàm mục tiêu (mất mát, regularization), sau đó tối ưu hóa nó!

Trong vòng lặp thứ t, chúng ta chỉ quan tâm đến hàm mục tiêu của vòng lặp này, và giả định rằng kết quả huấn luyện của vòng lặp t-1 đã được biết trước là y_i(t-1) . Do đó, hàm mục tiêu có thể được viết lại dưới dạng sau:
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/0a6f16c6-c67b-406f-bf84-5ce611a4d009)

Để tìm giá trị tối ưu của hàm mục tiêu tại thời điểm này tương đương với việc tìm giá trị tối ưu của f(x) là rất khó, vì vậy ta có thể sử dụng công triển khai taylor để xấp xỉ hàm này:
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/e99267fc-764d-4a0f-96d9-7037adc35021)

Lý do cho việc sử dụng giá trị y_i_t-1 là vì trong quá trình xây dựng chuỗi các cây quyết định, mỗi cây mới được huấn luyện để tối ưu hóa sai số còn lại giữa giá trị dự đoán và giá trị thật sự của y - tức là r_i_t = y_i - y_i_t-1. Do đó, khi xây dựng cây mới tại bước lặp t, chúng ta sẽ sử dụng giá trị y_i_t-1 làm đầu vào (input) để dự đoán sai số còn lại r_i_t, thay vì giá trị y_i ban đầu.
Vì vậy, khi xấp xỉ hàm mất mát, chúng ta sử dụng giá trị y_i_t-1 để tính toán sai số còn lại r_i_t

![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/4b84ff57-1923-456b-8aa8-6926d565d85c)

Và mục đích của ta sau khi xấp xỉ hàm này về một hàm dễ tính toán hơn thì ta có thể quan sát thấy tại bước t, ta đã biết giá trị y(t-1) nên l(y,y(t-1)) là 1 hằng số, mà hằng số thì không ảnh hưởng đến quá việc tối ưu hàm, nên ta có thể lược bỏ đi luôn:

![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/c31b1fd1-8a43-4a15-bbaa-d3faf6a36389)
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/9454ae60-42a0-4ef0-b75b-451a5628b3e1)

Vì độ phức tạp của cây quyết định có thể được xác định bởi số lượng nút lá , mô hình càng đơn giản khi có ít nút lá. Ngoài ra, các nút lá cũng không nên có trọng số cao (tương tự như trọng số của mỗi biến trong Logistic Regression).

Nói đơn giản, đối với mỗi mẫu dữ liệu, chúng ta sẽ phân chia nó vào một cái lá nhất định và gán cho lá đó một giá trị trọng số. Giá trị trọng số này biểu thị sự đóng góp của lá đó vào kết quả dự đoán của mẫu dữ liệu đó.
Để làm được điều này thì tác giả định nghĩa một hàm q(x) trong đó x là điểm dữ liệu và q(x) sẽ trả về vị trí của lá đó. ví dụ như:

![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/8fae663c-127c-4996-8efd-cc7db342ed31)

thì lá thứ nhất chứa mẫu {1,3}. Cụ thể hàm đó được định nghĩa như sau:
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/3dd58bb3-e3b1-4e0e-aadd-770189f418ca)

Như trên thì ta có thể viết lại là I_1 = {1,3}. Mục đích của biểu diễn này là tái tổ chức các mẫu dữ liệu dựa trên vị trí của các nút lá.
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/17a8e634-c895-4426-a868-5c354ce8bd0c)

vậy ta chỉ cần tối ưu w:
![image](https://github.com/tanaha2002/XGBoost_explain_and_example/assets/98084807/4c5e049a-e95c-4286-acc7-84ce16a7eb02)

Như đã nhấn mạnh ở trên, g_i và h_i được tính bằng cách lấy đạo hàm của y_i(t-1) theo dự đoán của mô hình. Do đó, chúng có thể coi là các hằng số.
Để tìm giá trị tối ưu của w, chúng ta cần tìm điểm mà đạo hàm của L^{(t)} đối với w bằng 0
