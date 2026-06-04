from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

MODEL_PATH = "risk_model.pkl"
COLUMNS_PATH = "model_columns.pkl"
CATEGORICAL_COLUMNS_PATH = "categorical_columns.pkl"

# 챗봇 서버 주소
# chatbot_server.py가 http://127.0.0.1:5001 에서 실행 중이어야 함
CHATBOT_API_URL = "http://127.0.0.1:5001/chat"

model = None
model_columns = None
categorical_columns = None


def load_artifacts():
    global model, model_columns, categorical_columns

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("risk_model.pkl 파일이 없습니다. 먼저 train_risk_model.py를 실행하세요.")

    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError("model_columns.pkl 파일이 없습니다. 먼저 train_risk_model.py를 실행하세요.")

    if not os.path.exists(CATEGORICAL_COLUMNS_PATH):
        raise FileNotFoundError("categorical_columns.pkl 파일이 없습니다. 먼저 train_risk_model.py를 실행하세요.")

    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    categorical_columns = joblib.load(CATEGORICAL_COLUMNS_PATH)

    if model is None:
        raise ValueError("모델 로드 실패")
    if model_columns is None:
        raise ValueError("컬럼 로드 실패")
    if categorical_columns is None:
        raise ValueError("범주형 컬럼 로드 실패")


def ensure_artifacts_loaded():
    global model, model_columns, categorical_columns
    if model is None or model_columns is None or categorical_columns is None:
        load_artifacts()


def preprocess_input(data):
    ensure_artifacts_loaded()

    required_fields = [
        "GENDER",
        "AGE_GROUP",
        "INCOME_GROUP",
        "FAMILY_TYPE",
        "MONTHLY_FEE_CODE",
        "HAS_AD_PLAN",
        "AD_INTENT",
        "USE_FREQUENCY",
        "USED_LAST_WEEK",
        "WEEKDAY_TIME_CODE",
        "WEEKEND_TIME_CODE",
        "DEVICE_COUNT",
        "OTT_SERVICE_COUNT",
        "CONTENT_TYPE_COUNT",
        "BROADCAST_TYPE_COUNT"
    ]

    missing_fields = [field for field in required_fields if field not in data or data[field] in [None, ""]]
    if missing_fields:
        raise ValueError(f"필수 입력값이 없습니다: {', '.join(missing_fields)}")

    input_dict = {
        "GENDER": str(data["GENDER"]),
        "AGE_GROUP": str(data["AGE_GROUP"]),
        "INCOME_GROUP": str(data["INCOME_GROUP"]),
        "FAMILY_TYPE": str(data["FAMILY_TYPE"]),
        "MONTHLY_FEE_CODE": str(data["MONTHLY_FEE_CODE"]),
        "HAS_AD_PLAN": str(data["HAS_AD_PLAN"]),
        "AD_INTENT": str(data["AD_INTENT"]),
        "USE_FREQUENCY": str(data["USE_FREQUENCY"]),
        "USED_LAST_WEEK": str(data["USED_LAST_WEEK"]),
        "WEEKDAY_TIME_CODE": str(data["WEEKDAY_TIME_CODE"]),
        "WEEKEND_TIME_CODE": str(data["WEEKEND_TIME_CODE"]),
        "DEVICE_COUNT": int(data["DEVICE_COUNT"]),
        "OTT_SERVICE_COUNT": int(data["OTT_SERVICE_COUNT"]),
        "CONTENT_TYPE_COUNT": int(data["CONTENT_TYPE_COUNT"]),
        "BROADCAST_TYPE_COUNT": int(data["BROADCAST_TYPE_COUNT"])
    }

    df = pd.DataFrame([input_dict])

    # 숫자형 보조 변환
    use_frequency_num = pd.to_numeric(df["USE_FREQUENCY"], errors="coerce").fillna(0)
    used_last_week_num = pd.to_numeric(df["USED_LAST_WEEK"], errors="coerce").fillna(0)
    weekday_time_num = pd.to_numeric(df["WEEKDAY_TIME_CODE"], errors="coerce").fillna(0)
    weekend_time_num = pd.to_numeric(df["WEEKEND_TIME_CODE"], errors="coerce").fillna(0)

    # 현재 UI에는 AVG_MIN_WEEKDAY, AVG_MIN_WEEKEND, SEARCH_VIEW, RECOMMEND_VIEW, BINGE_WATCH 직접 입력이 없으므로
    # 코드형 입력을 기반으로 근사 파생변수를 생성
    weekday_map = {1: 30, 2: 60, 3: 100, 4: 140}
    weekend_map = {1: 40, 2: 80, 3: 130, 4: 180}
    freq_engagement_map = {1: 4, 2: 7, 3: 10, 4: 13}

    weekday_avg_min = weekday_time_num.map(weekday_map).fillna(60)
    weekend_avg_min = weekend_time_num.map(weekend_map).fillna(80)

    df["WEEKLY_TOTAL_MIN"] = weekday_avg_min * 5 + weekend_avg_min * 2
    df["WEEKEND_RATIO"] = weekend_avg_min / (weekday_avg_min + weekend_avg_min + 1)
    df["ENGAGEMENT_SCORE"] = use_frequency_num.map(freq_engagement_map).fillna(7)
    df["ACTIVE_DAYS_SCORE"] = use_frequency_num + (3 - used_last_week_num)

    for col in model_columns:
        if col not in df.columns:
            if col in categorical_columns:
                df[col] = "0"
            else:
                df[col] = 0

    df = df[model_columns]

    for col in categorical_columns:
        df[col] = df[col].astype(str)

    for col in df.columns:
        if col not in categorical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sc1")
def sc1():
    return render_template("sc1.html")


@app.route("/sc2")
def sc2():
    return render_template("sc2.html")


# 실제 챗봇 사용 페이지
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        ensure_artifacts_loaded()

        data = request.get_json()
        if data is None:
            return jsonify({
                "success": False,
                "message": "JSON 데이터가 없습니다."
            }), 400

        input_df = preprocess_input(data)

        pred = model.predict(input_df)[0]
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            class_names = list(model.classes_)

            for i, class_name in enumerate(class_names):
                probabilities[str(class_name)] = round(float(proba[i]) * 100, 2)

        return jsonify({
            "success": True,
            "risk_group": str(pred),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# 프로젝트1에서 챗봇 서버로 질문을 전달하는 API
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

        response = requests.post(
            CHATBOT_API_URL,
            json={"question": question},
            timeout=180
        )

        if response.status_code != 200:
            return jsonify({
                "success": False,
                "message": f"챗봇 서버 오류: {response.status_code}"
            }), 500

        chatbot_result = response.json()

        return jsonify(chatbot_result)

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "message": "챗봇 서버에 연결할 수 없습니다. 먼저 chatbot_server.py가 실행 중인지 확인하세요."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "message": "챗봇 서버 응답 시간이 초과되었습니다. CPU 환경에서는 시간이 오래 걸릴 수 있습니다."
        }), 504

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


load_artifacts()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)