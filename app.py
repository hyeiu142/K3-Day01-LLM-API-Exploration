import os
import time
import streamlit as st
from openai import OpenAI

# Import helper functions from template
from template import (
    count_tokens,
    estimate_cost,
    retry_with_backoff,
    compare_models,
    PRICING_PER_1K_TOKENS,
    OPENAI_MODEL,
    OPENAI_MINI_MODEL,
)

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM API Playground",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Style all textareas and inputs to match dark mode and ensure text is highly readable */
    div[data-testid="stTextArea"] textarea, 
    div[data-testid="stTextInput"] input,
    div[data-testid="stChatInput"] textarea {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
    }
    
    /* Universal high contrast text for dark theme */
    .stApp p, .stApp span, .stApp div, .stApp li, .stMarkdown, .stMarkdown p {
        color: #F8FAFC;
    }

    /* Style Streamlit Chat Messages container and contents for high contrast */
    div[data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] strong,
    div[data-testid="stChatMessage"] em,
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] p {
        color: #FFFFFF !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* Labels styling to be bright and clear */
    label[data-testid="stWidgetLabel"] p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    
    /* Header gradient text */
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Stats badges */
    .stats-container {
        display: flex;
        gap: 15px;
        background: #1E293B;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-top: 5px;
        width: fit-content;
    }
    
    .stat-badge {
        font-size: 0.85rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .stat-badge-value {
        font-weight: 600;
        color: #34D399;
    }

    .stat-badge-value-cost {
        font-weight: 600;
        color: #38BDF8;
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0EA5E9, #10B981) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E293B;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Comparison table styling */
    .compare-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .compare-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 1px solid #475569;
        padding-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    
    .compare-body {
        font-size: 1rem;
        line-height: 1.6;
        color: #E2E8F0;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR / PARAMETERS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.markdown("### CẤU HÌNH API")
    
    # Custom API Key Input
    custom_key = st.text_input(
        "OpenAI API Key (Tùy chọn)",
        placeholder="sk-...",
        type="password",
        help="Để trống nếu muốn tự động lấy key cấu hình sẵn trong file .env",
    )
    
    if custom_key:
        os.environ["OPENAI_API_KEY"] = custom_key
    
    st.markdown("---")
    st.markdown("### THAM SỐ SINH TEXT")
    
    # Model Selection
    model_option = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o có trí thông minh vượt trội, gpt-4o-mini tối ưu chi phí và tốc độ."
    )
    
    # System Prompt (Persona)
    system_prompt = st.text_area(
        "System prompt - persona",
        value="Bạn là trợ lý lập trình thân thiện, trả lời ngắn gọn và cung cấp ví dụ code minh họa dễ hiểu bằng tiếng Việt.",
        height=100,
    )
    
    # Hyperparameters
    temp = st.slider("temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1, help="Độ ngẫu nhiên/sáng tạo.")
    top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.05, help="Giới hạn phân phối từ vựng.")
    max_tok = st.slider("max_tokens", min_value=16, max_value=2048, value=512, step=16, help="Giới hạn token output tối đa.")
    
    st.markdown("---")
    st.markdown("### ĐỘ BỀN & STREAMING")
    stream_enabled = st.checkbox("Streaming", value=True, help="Hiển thị phản hồi của AI mượt mà ngay khi được tạo ra.")
    retry_enabled = st.checkbox("Retry với backoff", value=True, help="Tự động thử lại chịu lỗi thông minh khi API quá tải.")
    
    st.markdown("---")
    if st.button("Xóa Lịch Sử Chat 🗑️"):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.markdown('<div class="header-title">LLM API Playground 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Giao diện trải nghiệm trực quan & thuyết trình các bài học của Lab 1</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chatbot Playground", "📊 So Sánh Model", "🧮 Tiktoken & Chi Phí"])

# Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# TAB 1: PLAYGROUND CHATBOT
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### 💬 Trợ Lý Tương Tác Đa Năng")
    st.write("Sử dụng cấu hình từ thanh bên (sidebar) để tương tác trực tiếp với mô hình.")
    
    # Display existing chat messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("stats"):
                stats = msg["stats"]
                st.markdown(
                    f"""
                    <div class="stats-container">
                        <div class="stat-badge">🔤 Tokens: <span class="stat-badge-value">in {stats['in']} / out {stats['out']}</span></div>
                        <div class="stat-badge">💸 Cost: <span class="stat-badge-value-cost">${stats['cost']:.5f}</span></div>
                        <div class="stat-badge">⚡ Latency: <span class="stat-badge-value">{stats['latency']:.2f}s</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Chat Input
    if user_input := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Generate Response
        with st.chat_message("assistant"):
            # Prepare messages list
            api_messages = [{"role": "system", "content": system_prompt}]
            # Add up to 3 turns (6 messages) of history
            history = st.session_state["messages"][-6:]
            for m in history:
                if m["role"] == "user":
                    api_messages.append({"role": "user", "content": m["content"]})
                elif m["role"] == "assistant":
                    api_messages.append({"role": "assistant", "content": m["content"]})

            # Check API Key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("⚠️ Không tìm thấy OpenAI API Key. Vui lòng nhập API Key ở sidebar để gọi API!")
                st.stop()
                
            client = OpenAI(api_key=api_key)
            
            # API Call logic with retry option
            def make_api_call():
                return client.chat.completions.create(
                    model=model_option,
                    messages=api_messages,
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=max_tok,
                    stream=stream_enabled,
                )
                
            try:
                start_time = time.time()
                
                if retry_enabled:
                    response_obj = retry_with_backoff(make_api_call, max_retries=3, base_delay=0.1)
                else:
                    response_obj = make_api_call()
                
                reply_content = ""
                
                # Render response
                if stream_enabled:
                    # Stream helper
                    def stream_generator():
                        for chunk in response_obj:
                            val = chunk.choices[0].delta.content or ""
                            yield val
                    reply_content = st.write_stream(stream_generator())
                else:
                    reply_content = response_obj.choices[0].message.content
                    st.markdown(reply_content)
                
                latency = time.time() - start_time
                
                # Estimate cost and tokens
                cost_info = estimate_cost(user_input, reply_content, model=model_option)
                
                stats_dict = {
                    "in": cost_info["input_tokens"],
                    "out": cost_info["output_tokens"],
                    "cost": cost_info["total_cost"],
                    "latency": latency
                }
                
                # Display metrics footer
                st.markdown(
                    f"""
                    <div class="stats-container">
                        <div class="stat-badge">🔤 Tokens: <span class="stat-badge-value">in {stats_dict['in']} / out {stats_dict['out']}</span></div>
                        <div class="stat-badge">💸 Cost: <span class="stat-badge-value-cost">${stats_dict['cost']:.5f}</span></div>
                        <div class="stat-badge">⚡ Latency: <span class="stat-badge-value">{stats_dict['latency']:.2f}s</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Save to history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": reply_content,
                    "stats": stats_dict
                })
                
            except Exception as e:
                st.error(f"Lỗi gọi API: {e}")

# ---------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 📊 So Sánh Song Song GPT-4o vs GPT-4o-mini")
    st.write("Nhập một câu hỏi, hệ thống sẽ gọi đồng thời cả hai model và lập bảng so sánh chi tiết chất lượng, độ trễ và chi phí.")
    
    compare_input = st.text_area(
        "Câu hỏi so sánh:",
        value="Việt Nam có bao nhiêu tỉnh thành? Hãy liệt kê 5 thành phố lớn trực thuộc trung ương kèm theo thế mạnh chính của mỗi thành phố.",
        height=80,
    )
    
    if st.button("Bắt đầu so sánh ⚡"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ Vui lòng nhập OpenAI API Key ở sidebar.")
        else:
            with st.spinner("Đang truy vấn cả hai mô hình..."):
                try:
                    # Run comparison
                    res = compare_models(compare_input)
                    
                    # Compute tokens and costs
                    cost_gpt4o = estimate_cost(compare_input, res["gpt4o_response"], model="gpt-4o")
                    cost_mini = estimate_cost(compare_input, res["mini_response"], model="gpt-4o-mini")
                    
                    # Columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="compare-card">
                                <div class="compare-header">
                                    <span>🤖 GPT-4o (Lớn)</span>
                                    <span style="color: #38BDF8;">${cost_gpt4o['total_cost']:.5f}</span>
                                </div>
                                <div class="compare-body">
                                    {res['gpt4o_response']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        # Metrics
                        st.metric("Thời gian phản hồi", f"{res['gpt4o_latency']:.2f} s")
                        st.metric("Tổng Token (In / Out)", f"{cost_gpt4o['input_tokens'] + cost_gpt4o['output_tokens']} ({cost_gpt4o['input_tokens']} / {cost_gpt4o['output_tokens']})")
                        
                    with col2:
                        st.markdown(
                            f"""
                            <div class="compare-card">
                                <div class="compare-header">
                                    <span>⚡ GPT-4o-mini (Nhỏ)</span>
                                    <span style="color: #34D399;">${cost_mini['total_cost']:.5f}</span>
                                </div>
                                <div class="compare-body">
                                    {res['mini_response']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        
                        # Metrics
                        st.metric("Thời gian phản hồi", f"{res['mini_latency']:.2f} s")
                        st.metric("Tổng Token (In / Out)", f"{cost_mini['input_tokens'] + cost_mini['output_tokens']} ({cost_mini['input_tokens']} / {cost_mini['output_tokens']})")
                    
                    # Summary alert
                    saving_factor = cost_gpt4o['total_cost'] / max(1e-9, cost_mini['total_cost'])
                    st.success(
                        f"👉 **Kết luận:** GPT-4o-mini hoàn thành với độ trễ tối ưu và **tiết kiệm chi phí gấp {saving_factor:.1f} lần** so với GPT-4o cho yêu cầu này!"
                    )
                    
                except Exception as e:
                    st.error(f"Lỗi so sánh: {e}")

# ---------------------------------------------------------------------------
# TAB 3: TOKEN COUNTER & COST ESTIMATOR
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### 🧮 Đếm Token & Ước Tính Chi Phí (Tiktoken)")
    st.write("Dán đoạn văn bản bất kỳ để xem tiktoken phân tách bao nhiêu token và so sánh với phương pháp ước lượng số từ thông thường.")
    
    text_to_count = st.text_area(
        "Nhập văn bản cần phân tích:",
        value="Blockchain là một công nghệ lưu trữ dữ liệu dưới dạng sổ cái phân tán. Mọi thông tin được liên kết với nhau bằng mã hóa mật mã và không thể thay đổi một khi đã ghi nhận.",
        height=150,
    )
    
    if text_to_count:
        # standard word count
        words = len(text_to_count.split())
        approx_tokens = words / 0.75
        
        # Real token count via tiktoken
        tokens_4o = count_tokens(text_to_count, model="gpt-4o")
        tokens_mini = count_tokens(text_to_count, model="gpt-4o-mini")
        
        # Layout
        c1, c2, c3 = st.columns(3)
        c1.metric("Số từ (Words)", f"{words} từ")
        c2.metric("Token ước lượng (Words / 0.75)", f"{approx_tokens:.1f} tokens")
        c3.metric("Token thực tế (Tiktoken)", f"{tokens_4o} tokens")
        
        st.markdown("---")
        st.markdown("#### Bảng Ước Tính Chi Phí Nếu Văn Bản Này Là:")
        
        pricing_4o = PRICING_PER_1K_TOKENS["gpt-4o"]
        pricing_mini = PRICING_PER_1K_TOKENS["gpt-4o-mini"]
        
        in_cost_4o = (tokens_4o / 1000) * pricing_4o["input"]
        out_cost_4o = (tokens_4o / 1000) * pricing_4o["output"]
        
        in_cost_mini = (tokens_mini / 1000) * pricing_mini["input"]
        out_cost_mini = (tokens_mini / 1000) * pricing_mini["output"]
        
        # Display comparison table
        st.table([
            {
                "Vai trò văn bản": "Input (Prompt)",
                "Chi phí GPT-4o": f"${in_cost_4o:.6f}",
                "Chi phí GPT-4o-mini": f"${in_cost_mini:.6f}",
                "Tỷ lệ chênh lệch": f"{pricing_4o['input'] / pricing_mini['input']:.1f}x"
            },
            {
                "Vai trò văn bản": "Output (Phản hồi)",
                "Chi phí GPT-4o": f"${out_cost_4o:.6f}",
                "Chi phí GPT-4o-mini": f"${out_cost_mini:.6f}",
                "Tỷ lệ chênh lệch": f"{pricing_4o['output'] / pricing_mini['output']:.1f}x"
            }
        ])
        
        st.info("💡 Lưu ý: Đối với tiếng Việt có dấu, số lượng token thực tế thường nhiều hơn số từ đáng kể (khoảng gấp 1.5 - 2 lần) do tokenizer mã hóa các ký tự đặc biệt.")
