# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần từ 0.0 đến 1.5, phản hồi của mô hình chuyển từ tính ổn định, nhất quán và dễ đoán sang sự sáng tạo, đa dạng từ vựng. Ở mức 0.0 và 0.5, câu trả lời rất mạch lạc và tập trung vào các sự thật lịch sử/địa lý quen thuộc. Tuy nhiên, ở mức 1.5, văn bản bắt đầu trở nên lộn xộn, thiếu cấu trúc rõ ràng và có thể xuất hiện các câu chữ kỳ lạ hoặc lặp từ không kiểm soát.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Cho chatbot hỗ trợ khách hàng, ta nên đặt temperature ở mức thấp (từ 0.0 đến 0.2). Lý do là vì hệ thống chăm sóc khách hàng cần sự chính xác tuyệt đối, thông tin nhất quán và tránh tối đa việc mô hình "sáng tạo" quá đà hoặc sinh ra thông tin sai sự thật (hallucination).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Dựa vào bảng giá (gpt-4o có giá input $0.0025, output $0.010; gpt-4o-mini có giá input $0.00015, output $0.0006), GPT-4o đắt hơn GPT-4o-mini đúng 16.67 lần (khoảng 17 lần) cho cả chi phí đầu vào và đầu ra. 
> - **Trường hợp GPT-4o xứng đáng:** Cần xử lý các tác vụ phức tạp như phân tích logic lập trình, giải quyết bài toán toán học chuyên sâu, hoặc phân tích báo cáo tài chính đa chiều yêu cầu khả năng suy luận vượt trội.
> - **Trường hợp nên dùng GPT-4o-mini:** Chatbot hỗ trợ khách hàng thông thường, tóm tắt văn bản ngắn, phân loại ý kiến người dùng (sentiment analysis), hoặc các tác vụ dịch thuật đơn giản nơi tốc độ phản hồi nhanh và chi phí thấp là ưu tiên hàng đầu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi có sự khác biệt rõ rệt về phong cách và đối tượng hướng đến: giáo viên tiểu học dùng câu từ ngắn gọn, ví dụ trực quan như "cuốn sổ chung của lớp học" và không dùng thuật ngữ kỹ thuật, trong khi chuyên gia tài chính sử dụng từ vựng học thuật phức tạp như "cơ chế đồng thuận", "sổ cái phân tán", "mã hóa bất đối xứng". System prompt có vai trò định hình sâu sắc hành vi của mô hình bằng cách giới hạn miền tri thức, quy định tông giọng (tone of voice) và cấu trúc nội dung phù hợp với đối tượng mục tiêu. Điều này giúp kiểm soát chất lượng đầu ra mà không cần thay đổi câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, ước tính theo công thức `100 / 0.75 = 133` tokens, trong khi tiktoken đếm thực tế khoảng 160-180 tokens (chênh lệch từ 20% đến 35%). Tiếng Việt tốn nhiều token hơn tiếng Anh có cùng độ dài vì các thuật toán phân tách token (như cl100k_base của OpenAI) được tối ưu hóa chủ yếu cho tiếng Anh và các ngôn ngữ hệ Latin không dấu; khi gặp tiếng Việt với các ký tự có dấu (UTF-8 multi-byte), tokenizer phải chia nhỏ từ thành các mảnh con (sub-words) hoặc thậm chí là từng bytes ký tự lẻ, khiến số lượng token tăng vọt.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng cuối (như chatbot, trợ lý ảo) để hiển thị phản hồi ngay lập tức (giảm thiểu Time to First Token), tạo cảm giác mượt mà và giảm sự sốt ruột cho người dùng khi mô hình sinh câu trả lời dài. Ngược lại, non-streaming phù hợp hơn khi chạy các tác vụ ngầm (background jobs), xử lý dữ liệu hàng loạt (batch processing), hoặc khi cần kết quả hoàn chỉnh để thực hiện các bước xử lý logic tiếp theo (như trích xuất JSON có cấu trúc để nạp vào cơ sở dữ liệu).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có lợi thế lớn là giảm tải dần áp lực lên server đang bị quá tải bằng cách kéo giãn thời gian chờ giữa các lần thử lại sau mỗi thất bại (0.1s -> 0.2s -> 0.4s...). Nếu hàng nghìn client cùng retry với một khoảng delay cố định như nhau, chúng sẽ tạo ra hiện tượng "thundering herd problem" (hiệu ứng bầy đàn), tiếp tục đồng loạt gửi yêu cầu cực lớn vào server cùng một thời điểm, khiến server bị nghẽn liên tục và không thể phục hồi được.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
> - Lựa chọn cụm từ "trả lời ngắn gọn" là rất quan trọng để tối ưu hóa chi phí API (tiết kiệm token output) và giúp người dùng đọc thông tin nhanh chóng trên môi trường CLI thay vì nhận các câu trả lời dài dòng.
> - Việc chỉ định "bằng tiếng Việt" đảm bảo tính đồng nhất về ngôn ngữ phục vụ cho sinh viên Việt Nam, tránh việc mô hình tự ý dịch hoặc phản hồi bằng tiếng Anh khi gặp các thuật ngữ công nghệ tiếng Anh trong câu hỏi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là chatbot chỉ ghi nhớ được 3 lượt hội thoại gần nhất (nhằm tiết kiệm token đầu vào), dẫn đến việc mô hình sẽ nhanh chóng quên mất các ngữ cảnh cũ hơn của cuộc hội thoại dài. 
> - **Đề xuất cải thiện:** Triển khai cơ chế tóm tắt lịch sử hội thoại (Conversation Summary Memory).
> - **Mô tả cách triển khai:** Khi lịch sử vượt quá 3 lượt, thay vì cắt bỏ hoàn toàn các tin nhắn cũ, ta sẽ gọi một API chạy ngầm nhờ mô hình tóm tắt lại các điểm chính của cuộc trò chuyện từ trước đến nay thành một đoạn tóm tắt ngắn (summary). Đoạn tóm tắt này sau đó luôn được chèn vào system prompt hoặc message đầu tiên cùng với system prompt của trợ lý. Điều này giúp mô hình vẫn giữ được các ý chính xuyên suốt mà không làm tăng quá nhiều số lượng token của các message chi tiết.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
