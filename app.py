import random
import streamlit as st
import os, json, uuid, requests, logging, random

from dotenv import load_dotenv, find_dotenv
from typing import Annotated, TypedDict, List
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# ────────────────────────────────────────────────
# 1. 환경 변수 로딩
# ────────────────────────────────────────────────
@st.cache_resource
def load_env():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    return {
        "openai":   os.getenv("OPENAI_API_KEY", ""),
        "pixabay":  os.getenv("PIXABAY_API_KEY", ""),
        "kakao":    os.getenv("KAKAO_OAUTH_TOKEN", ""),
    }

ENV = load_env()

# ────────────────────────────────────────────────
# 2. LangChain LLM & Tools
# ────────────────────────────────────────────────
MODEL = "gpt-4o"

@tool
def send_kakao_alert(message: str) -> bool:
    """카카오톡으로 주변인에게 경고 메시지를 전송한다.

    :param token: Kakao OAuth 토큰
    :param message: 전송할 메시지 텍스트
    :return: 성공 여부
    """
    
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers={
        "Authorization" : "Bearer " + ENV["kakao"],
        "Content-Type": "application/x-www-form-urlencoded"
    }
    # template_object는 카카오톡 메시지 형식을 JSON 문자열로 전달
    template = {
        "object_type": "text",
        "text": message,
        "link": {
            "web_url": "https://www.google.com",  # 임의로 정의함
            "mobile_web_url": "https://www.google.com",
        },
        "button_title": "도움 받기"
    }
    data = {"template_object": json.dumps(template, ensure_ascii=False)}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code == 200:
            return True
        logging.warning("Kakao alert failed with status %s: %s", response.status_code, response.text)
    except Exception as exc:
        logging.warning("Exception while sending Kakao alert: %s", exc)
    return False


@tool
def show_cute_cat_image() -> str:
    """마음의 안정을 위해 귀여운 고양이 사진을 랜덤으로 보여준다."""
    if not ENV["pixabay"]:
        return "PIXABAY_KEY_MISSING"
    
    cat_keywords = [
        "cute+cat", "kitten", "cat+playing", "fluffy+cat", 
        "tabby+cat", "cat+sleeping", "peaceful+cat"
    ]
    keyword = random.choice(cat_keywords)
    
    pix_url = (f"https://pixabay.com/api/?key={ENV['pixabay']}"
               f"&q={keyword}&image_type=photo&per_page=20")
    try:
        data = requests.get(pix_url, timeout=10).json()
        if data["totalHits"] == 0:
            return "NO_CAT_FOUND"
        
        random_cat = random.choice(data['hits'])
        return f"CAT_IMAGE_URL:{random_cat['webformatURL']}"
    except Exception as e:
        logging.exception(e)
        return f"PIXABAY_ERROR:{e}"

TOOLS = [send_kakao_alert, show_cute_cat_image, TavilySearch(max_results=3)]
LLM = ChatOpenAI(model=MODEL, temperature=0.1, max_tokens=1024)

# ────────────────────────────────────────────────
# 3. Streamlit UI 기본 설정
# ────────────────────────────────────────────────
st.set_page_config(page_title="AI 심리상담 챗봇", layout="wide")

if ENV["kakao"]:
        try:
            # 카카오톡 연결 테스트
            url = "https://kapi.kakao.com/v1/user/access_token_info"
            headers = {"Authorization": f"Bearer {ENV['kakao']}"}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                # 연결 성공 시 구동 알림 전송
                startup_message = f"""AI 심리상담 챗봇 구동 성공"""
                
                send_result = send_kakao_alert.invoke(startup_message)
        except Exception as e:
            print(f"AI 심리상담 챗봇 구동 실패: {e}")

# ────────────────────────────────────────────────
# 4. 심리상담 전문 프롬프트
# ────────────────────────────────────────────────
COUNSELING_PROMPT = ChatPromptTemplate.from_messages([  # 생성형  AI로 프롬프트 작성함
    ("system", """당신은 전문적이고 따뜻한 심리상담 AI입니다. 다음 원칙들을 따라 상담을 진행해주세요:

**위험상황 감지 및 대응 (최우선):**
- 사용자가 자살, 자해, 극도의 절망감을 표현할 때는 IMMEDIATELY send_kakao_alert 도구를 사용
- 다음과 같은 표현을 감지하면 사용자의 의사와 무관하게 즉시 경고 발송:
  * "죽고 싶다", "자살하고 싶다", "사라지고 싶다"
  * "자해하고 싶다", "상처내고 싶다"
  * "더 이상 견딜 수 없다", "포기하고 싶다"
  * "살 이유가 없다", "의미가 없다"
- 경고 발송 후에도 따뜻하고 지지적인 응답 제공
     
**상담 원칙:**
1. **경청과 공감**: 사용자의 감정을 진심으로 이해하고 공감하며, 판단하지 않는 태도를 유지합니다.
2. **안전한 환경 조성**: 사용자가 안전하다고 느끼도록 따뜻하고 지지적인 분위기를 만듭니다.
3. **적극적 경청**: 사용자의 말 속에 숨겨진 감정과 니즈를 파악하고 반영해줍니다.
4. **단계별 접근**: 급하게 해결책을 제시하기보다는 먼저 감정을 충분히 탐색합니다.

**상황별 대응:**
- **정서적 고통 및 위험 징후 (자해, 자살 등)**: 사용자의 의사와 무관하게 즉시 send_kakao_alert 도구로 주변인 개입 요청
- **일반적 스트레스/우울**: 감정 탐색 → 공감적 지지 → 대처방안 함께 모색 또는 show_cute_cat_image 도구로 마음의 안정 도움
- **정보가 필요한 경우**: TavilySearch로 신뢰할 수 있는 정보 제공

**대화 스타일:**
- 존댓말 사용, 따뜻하고 차분한 톤
- 열린 질문으로 더 깊은 탐색 유도
- "어떤 기분이셨을까요?", "더 자세히 말씀해주실 수 있나요?" 같은 격려
- 사용자의 강점과 자원을 발견하고 인정

**주의사항:**
- 의학적 진단이나 처방은 절대 하지 않음
- 전문 치료가 필요한 경우 전문기관 연계 권유
- 사용자의 속도에 맞춰 진행

이전 대화 맥락을 기억하고 연속성 있는 상담을 제공해주세요."""),
    
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ────────────────────────────────────────────────
# 5. LangGraph 상태 정의 & 실행 노드
# ────────────────────────────────────────────────
class ChatState(TypedDict):
    input: str
    response: str
    messages: Annotated[List, add_messages]

def run_agent(state: ChatState):
    user_input = state["input"]
    history_msgs = state.get("messages", [])
    
    # 현재 사용자 메시지를 히스토리에 추가
    msgs_for_agent = history_msgs + [HumanMessage(content=user_input)]

    # 에이전트 호출
    agent_response = AGENT.invoke({
        "input": user_input,
        "messages": msgs_for_agent
    })

    ai_msg = agent_response["messages"][-1]
    updated_msgs = msgs_for_agent + [ai_msg]

    return {
        "response": ai_msg.content, 
        "messages": updated_msgs
    }

# AGENT 생성
AGENT = create_react_agent(LLM, TOOLS, state_modifier=COUNSELING_PROMPT, verbose=True)

# 그래프 구성
graph = StateGraph(ChatState)
graph.add_node("chat", run_agent)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
APP = graph.compile(checkpointer=MemorySaver())

# ────────────────────────────────────────────────
# 6. Streamlit 세션 스테이트
# ────────────────────────────────────────────────
if "lc_msgs" not in st.session_state:
    st.session_state.lc_msgs = []
if "cat_imgs" not in st.session_state:
    st.session_state.cat_imgs = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ────────────────────────────────────────────────
# 7. 사이드바
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    .status-right {
        text-align: right;
        color: #888; 
        font-size: 0.7em;
    }
    .error-right {
        text-align: right;
        color: #ff4444;  /* 빨간색 */
        font-size: 0.7em;
        font-weight: bold;
    }
    .feature-header {
        font-size: 1.8em !important;
        font-weight: bold !important;
        margin-top: 5px !important;
        margin-bottom: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if ENV['openai']:
        if ENV['pixabay']:
            if ENV['kakao']:
                 st.markdown("<div class='status-right'>정상 작동 중</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-right'>카카오톡 API 오류</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-right'>Pixabay API 오류</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='error-right'>OpenAI API 오류</div>", unsafe_allow_html=True)
   
    st.markdown("""
    <div style='margin-top: -30px; margin-bottom: -15px;'>
        <hr style='margin: 10px 0;'>
    </div>
    
    <h2 style='margin-top: 5px; margin-bottom: 15px; font-size: 1.8em;'>특징</h2>
    
    <div style='margin-bottom: 10px;'>
        <strong>• 휘발성</strong><br>
        진솔한 상담을 위해 상담 내용을 저장하지 않습니다
    </div>
    
    <div style='margin-bottom: 10px;'>
        <strong>• 프라이빗</strong><br>
        당신의 상담 내용은 외부에 공개되지 않습니다
    </div>
    
    <div style='margin-bottom: 10px;'>
        <strong>• 결정안함</strong><br>
        필요시 전문기관을 안내해드립니다
    </div>
    
    <div style='margin-bottom: 40px;'>
        <strong>• 도움요청</strong><br>
        필요시 주변인에게 도움을 요청할 수 있습니다<br>
        <small style='color: #666; font-style: italic; font-size: 0.8em;'>(현재는 '나에게 보내기')</small>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("대화 초기화"):
        st.session_state.lc_msgs = []
        st.session_state.cat_imgs = []
        st.session_state.thread_id = str(uuid.uuid4())
        if "welcome_shown" in st.session_state:
            del st.session_state.welcome_shown
        st.rerun()

# ────────────────────────────────────────────────
# 8. 메인 페이지 헤더
# ────────────────────────────────────────────────
st.title("AI 심리상담 챗봇")
st.markdown("---")

# ────────────────────────────────────────────────
# 9. 채팅 스타일 CSS
# ────────────────────────────────────────────────
st.markdown("""
<style>
/* 메인 컨테이너 스타일 */
.main-chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

/* 사용자 메시지 (우측 정렬) */
.user-message {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    margin: 15px 0;
    padding-left: 20%;
}

.user-bubble {
    background: #e3f2fd;
    border: 1px solid #2196f3;
    border-radius: 18px 18px 5px 18px;
    padding: 12px 16px;
    max-width: 80%;
    margin-left: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.user-avatar {
    font-size: 24px;
    margin-left: 8px;
    margin-top: 2px;
}

/* 챗봇 메시지 (좌측 정렬) */
.bot-message {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    margin: 15px 0;
    padding-right: 20%;
}

.bot-bubble {
    background: #fff8e1;
    border: 1px solid #ff9800;
    border-radius: 18px 18px 18px 5px;
    padding: 12px 16px;
    max-width: 80%;
    margin-right: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.bot-avatar {
    font-size: 24px;
    margin-right: 8px;
    margin-top: 2px;
}

/* 메시지 텍스트 스타일 */
.message-text {
    line-height: 1.6;
    word-wrap: break-word;
    font-size: 14px;
}

/* 이미지 컨테이너 */
.image-container {
    margin-top: 10px;
    text-align: center;
}

/* 채팅 입력창 스타일 수정 */
.stChatInput {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 10px 0;
    border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

def render_custom_message(role: str, content: str, img_url: str = None):
    """커스텀 메시지 렌더링"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">
                <div class="message-text">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:  # assistant
        st.markdown(f"""
        <div class="bot-message">
            <div class="bot-bubble">
                <div class="message-text">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 고양이 이미지가 있으면 표시
        if img_url:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img_url, caption="마음의 평안을 위한 귀여운 고양이", width=300)
            st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
# 10. 대화 기록 렌더링
# ────────────────────────────────────────────────
# 최초 메시지 처리
if not st.session_state.lc_msgs and "welcome_shown" not in st.session_state:
    welcome_msg = """안녕하세요. 저는 AI 심리상담 챗봇입니다. 이곳은 당신을 위한 공간입니다. 어떤 이야기든 편안하게 나눠주세요. 오늘 기분은 어떠신가요?"""
    
    st.session_state.lc_msgs.append(AIMessage(content=welcome_msg))
    st.session_state.welcome_shown = True

# 대화 기록 표시
st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
for idx, msg in enumerate(st.session_state.lc_msgs):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    img_url = None
    
    # 고양이 이미지 URL 확인
    if role == "assistant" and idx < len(st.session_state.cat_imgs):
        img_url = st.session_state.cat_imgs[idx]
    
    render_custom_message(role, msg.content, img_url)
st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
# 11. 사용자 입력 처리
# ────────────────────────────────────────────────
if user_text := st.chat_input("입력"):
    # 사용자 메시지 표시
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    render_custom_message("user", user_text)
    st.markdown('</div>', unsafe_allow_html=True)

    init_state = {
        "input": user_text,
        "response": "",
        "messages": st.session_state.lc_msgs.copy()
    }

    cfg = RunnableConfig(
        recursion_limit=50,
        configurable={"thread_id": st.session_state.thread_id}
    )

    # 봇 응답 처리
    with st.spinner("생각 중..."):
        try:
            final_state = APP.invoke(init_state, config=cfg)
            bot_reply = final_state["response"]
            
            # 고양이 이미지 URL 추출
            img_url = None
            if "CAT_IMAGE_URL:" in bot_reply:
                img_url = bot_reply.split("CAT_IMAGE_URL:")[1].split()[0]
                st.session_state.cat_imgs.append(img_url)
                bot_reply = bot_reply.replace(f"CAT_IMAGE_URL:{img_url}", "").strip()
            
            # 봇 메시지 표시
            st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
            render_custom_message("assistant", bot_reply, img_url)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 세션 상태 업데이트
            st.session_state.lc_msgs = final_state["messages"]
            if st.session_state.lc_msgs:
                st.session_state.lc_msgs[-1] = AIMessage(content=bot_reply)
                
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
            render_custom_message("assistant", "죄송합니다. 일시적인 문제가 발생했습니다. 다시 말씀해주시겠어요?")
            st.markdown('</div>', unsafe_allow_html=True)

    # 페이지 새로고침으로 새 메시지 표시
    st.rerun()
