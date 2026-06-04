import re
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./outputs/ott-code-assistant-adapter"

SYSTEM_PROMPT = """
너는 OTT Retention Analytics 프로젝트 전용 코드 설명 및 오류 해결 챗봇이다.

반드시 지켜야 할 규칙:
1. 답변은 반드시 자연스러운 한국어로만 작성한다.
2. 중국어 한자, 일본어, 이상한 외국어 표현, 불필요한 영어 혼용을 절대 섞지 않는다.
3. 질문이 프로젝트 코드, 파일, API, 오류 해결과 관련되지 않으면 프로젝트 질문을 유도한다.
4. 질문이 너무 짧거나 의미가 불명확하면 추측하지 말고 구체적인 질문을 요청한다.
5. 모르는 내용은 지어내지 말고 "현재 제공된 프로젝트 기준으로는 확인되지 않습니다"라고 말한다.
6. 파일 역할을 절대 헷갈리지 않는다.
7. 답변은 먼저 핵심부터 말하고, 그다음 관련 파일명과 흐름을 설명한다.

프로젝트 파일 역할:
- app.py:
  Flask 서버 파일이다.
  /, /sc1, /sc2, /chatbot 화면 라우터를 제공한다.
  /predict API에서 사용자의 입력값을 받아 CatBoost 모델로 안전/주의/위험 이탈 위험군을 예측한다.
  risk_model.pkl, model_columns.pkl, categorical_columns.pkl을 로드한다.
  /chat API를 통해 챗봇 서버와 통신한다.

- train_risk_model.py:
  Oracle DB에서 데이터를 불러와 모델 학습용 데이터를 만든다.
  USERS, ECONOMY, SUBSCRIPTION, USAGE_BEHAVIOR 및 복수응답 테이블을 조인한다.
  파생변수를 생성하고 CatBoostClassifier로 안전/주의/위험 3-class 모델을 학습한다.
  학습 결과로 risk_model.pkl, model_columns.pkl, categorical_columns.pkl을 저장한다.

- load_to_oracle.py:
  CSV 또는 설문 데이터를 Oracle DB에 적재하는 스크립트다.
  Flask 라우터를 정의하지 않는다.
  /predict API를 처리하지 않는다.
  예측 결과를 반환하지 않는다.

- sc1.html:
  이탈 위험군 예측 입력 화면이다.
  사용자가 선택한 값을 payload로 만들고 fetch('/predict')로 app.py에 POST 요청을 보낸다.
  응답받은 risk_group과 probabilities를 결과 모달에 표시한다.

- sc2.html:
  Base Model과 QLoRA 학습 후 모델의 답변 차이를 보여주는 학습 전후 비교 페이지다.

- chatbot.html:
  실제 AI 챗봇을 사용하는 페이지다.
  사용자의 질문을 /chat API로 보내고 챗봇 답변을 화면에 표시한다.

좋은 답변 예시:
질문: /predict 라우터는 뭐 하는 코드야?
답변: 이 프로젝트의 /predict 라우터는 sc1.html에서 전달한 JSON 데이터를 받아 CatBoost 모델로 이탈 위험군을 예측하는 API입니다. 입력값은 preprocess_input() 함수에서 모델 입력 형태로 변환되고, risk_model.pkl을 사용해 안전/주의/위험 결과와 클래스별 확률을 반환합니다.
"""


app = Flask(__name__)
CORS(app)


def is_greeting(question):
    question = question.strip().lower()

    greetings = {
        "안녕",
        "안녕하세요",
        "하이",
        "ㅎㅇ",
        "hi",
        "hello",
        "헬로",
        "반가워",
        "반갑습니다"
    }

    return question in greetings


def greeting_response():
    return (
        "안녕하세요! 저는 OTT Retention Analytics 프로젝트 전용 코드 챗봇입니다.\n\n"
        "다음과 같은 질문을 할 수 있습니다.\n"
        "- /predict 라우터는 뭐 하는 코드야?\n"
        "- sc1.html에서 예측 실행 버튼을 누르면 어떤 흐름으로 동작해?\n"
        "- risk_model.pkl 파일이 없다는 오류가 뜨면 어떻게 해야 해?\n"
        "- train_risk_model.py와 load_to_oracle.py 차이를 설명해줘."
    )


def is_meaningless_question(question):
    question = question.strip()

    if len(question) < 5:
        return True

    meaningless_words = {
        "아",
        "어",
        "ㅇ",
        "ㅇㅇ",
        "ㄱ",
        "ㄱㄱ",
        "ㅎ",
        "ㅎㅎ",
        "테스트",
        "test"
    }

    if question.lower() in meaningless_words:
        return True

    return False


def meaningless_response():
    return (
        "질문이 너무 짧아 의도를 파악하기 어렵습니다.\n\n"
        "아래처럼 프로젝트와 관련된 구체적인 질문을 입력해 주세요.\n"
        "- /predict 라우터는 뭐 하는 코드야?\n"
        "- sc1.html에서 예측 실행 버튼을 누르면 어떤 흐름으로 동작해?\n"
        "- risk_model.pkl 파일이 없다는 오류가 뜨면 어떻게 해야 해?\n"
        "- load_to_oracle.py와 train_risk_model.py 차이를 설명해줘."
    )


def is_project_related_question(question):
    q = question.lower()

    project_keywords = [
        "app.py",
        "train_risk_model",
        "load_to_oracle",
        "sc1",
        "sc2",
        "chatbot",
        "index.html",
        "intro.css",
        "sc1.css",
        "sc2.css",
        "/predict",
        "/chat",
        "predict",
        "preprocess_input",
        "risk_model",
        "model_columns",
        "categorical_columns",
        "catboost",
        "oracle",
        "flask",
        "json",
        "fetch",
        "payload",
        "라우터",
        "라우트",
        "모델",
        "파일",
        "오류",
        "에러",
        "예측",
        "이탈",
        "위험군",
        "학습",
        "데이터",
        "컬럼",
        "전처리",
        "함수",
        "코드",
        "구조",
        "역할",
        "설명",
        "흐름"
    ]

    return any(keyword in q for keyword in project_keywords)


def not_project_related_response():
    return (
        "저는 일반 대화용 챗봇이 아니라, OTT Retention Analytics 프로젝트 전용 코드 챗봇입니다.\n\n"
        "프로젝트 코드나 오류 해결과 관련된 질문을 입력해 주세요.\n"
        "예시:\n"
        "- app.py는 무슨 역할이야?\n"
        "- /predict 라우터는 뭐 하는 코드야?\n"
        "- sc1.html에서 예측 버튼을 누르면 어떤 흐름으로 동작해?\n"
        "- risk_model.pkl 파일이 없으면 어떻게 해야 해?"
    )


def contains_bad_cjk(text):
    """
    한글은 허용하고, 중국어 한자/일본어 문자만 감지한다.
    """
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", text))


def clean_bad_cjk(text):
    """
    출력에 섞인 중국어 한자/일본어 문자를 제거한다.
    단, 한글은 제거하지 않는다.
    """
    text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" .", ".").replace(" ,", ",")
    return text.strip()


def clean_mixed_english(text):
    """
    자주 튀는 이상한 혼용 표현을 정리한다.
    """
    replacements = {
        "covered해줘": "설명해줘",
        "cover해줘": "설명해줘",
        "covered": "설명",
        "routing": "라우팅",
        "flask": "Flask",
        "oracle": "Oracle",
        "catboost": "CatBoost",
        "api": "API",
        "json": "JSON"
    }

    for before, after in replacements.items():
        text = text.replace(before, after)

    return text.strip()


def postprocess_answer(answer):
    answer = answer.strip()

    if contains_bad_cjk(answer):
        answer = clean_bad_cjk(answer)

    answer = clean_mixed_english(answer)

    if len(answer) < 10:
        answer = meaningless_response()

    return answer


print("토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Base Model 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None,
    trust_remote_code=True
)


print("LoRA Adapter 로드 중...")
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()
print("챗봇 모델 로드 완료")


def build_prompt(question):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def extract_answer(decoded):
    if "<|im_start|>assistant" in decoded:
        answer = decoded.split("<|im_start|>assistant")[-1]
        answer = answer.replace("<|im_end|>", "").strip()
    else:
        answer = decoded.strip()

    return answer


def generate_answer(question):
    question = question.strip()

    # 1. 인사는 모델에 보내지 않고 서버에서 바로 처리
    if is_greeting(question):
        return greeting_response()

    # 2. 의미 없는 짧은 입력도 모델에 보내지 않음
    if is_meaningless_question(question):
        return meaningless_response()

    # 3. 프로젝트와 너무 관련 없는 질문도 서버에서 안내
    if not is_project_related_question(question):
        return not_project_related_response()

    # 4. 여기부터는 실제 QLoRA 모델에게 질문
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=False,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = extract_answer(decoded)
    answer = postprocess_answer(answer)

    return answer


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "success": False,
                "message": "JSON 데이터가 없습니다."
            }), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "success": False,
                "message": "질문을 입력하세요."
            }), 400

        answer = generate_answer(question)

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "챗봇 서버 정상 동작 중"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)