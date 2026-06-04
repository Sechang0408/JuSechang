import pandas as pd
import oracledb
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Oracle 연결 설정
DB_USER = "scott"
DB_PASSWORD = "tiger"
DB_HOST = "localhost"
DB_PORT = 1521
DB_SERVICE = "xe"

ORACLE_CLIENT_PATH = r"C:\oraclexe\app\oracle\product\11.2.0\server\bin"


# 2. Oracle 연결
conn = None

try:
    try:
        oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_PATH)
    except Exception:
        pass

    conn = oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        service_name=DB_SERVICE
    )

    print("Oracle 연결 성공")
    print("thin mode 여부:", conn.thin)

    # 3. 기본 데이터 조회
    base_query = """
    SELECT
        u.USER_SEQ,
        u.GENDER,
        u.AGE_GROUP,
        e.INCOME_GROUP,
        e.FAMILY_TYPE,
        s.MONTHLY_FEE_CODE,
        s.HAS_AD_PLAN,
        s.AD_INTENT,
        ub.USE_FREQUENCY,
        ub.WEEKDAY_TIME_CODE,
        ub.WEEKEND_TIME_CODE,
        ub.USED_LAST_WEEK,
        ub.AVG_MIN_WEEKDAY,
        ub.AVG_MIN_WEEKEND,
        ub.SEARCH_VIEW,
        ub.RECOMMEND_VIEW,
        ub.BINGE_WATCH
    FROM USERS u
    JOIN ECONOMY e
        ON u.USER_SEQ = e.USER_SEQ
    JOIN SUBSCRIPTION s
        ON u.USER_SEQ = s.USER_SEQ
    JOIN USAGE_BEHAVIOR ub
        ON u.USER_SEQ = ub.USER_SEQ
    """

    df = pd.read_sql(base_query, conn)
    print("기본 데이터 shape:", df.shape)
    print(df.head())

    # 4. 복수응답 테이블 집계
    q_device = """
    SELECT USER_SEQ, COUNT(*) AS DEVICE_COUNT
    FROM USER_DEVICE
    GROUP BY USER_SEQ
    """

    q_service = """
    SELECT USER_SEQ, COUNT(*) AS OTT_SERVICE_COUNT
    FROM USER_OTT_SERVICE
    GROUP BY USER_SEQ
    """

    q_content = """
    SELECT USER_SEQ, COUNT(*) AS CONTENT_TYPE_COUNT
    FROM USER_CONTENT_TYPE
    GROUP BY USER_SEQ
    """

    q_broadcast = """
    SELECT USER_SEQ, COUNT(*) AS BROADCAST_TYPE_COUNT
    FROM USER_BROADCAST_TYPE
    GROUP BY USER_SEQ
    """

    device_df = pd.read_sql(q_device, conn)
    service_df = pd.read_sql(q_service, conn)
    content_df = pd.read_sql(q_content, conn)
    broadcast_df = pd.read_sql(q_broadcast, conn)

    df = df.merge(device_df, on="USER_SEQ", how="left")
    df = df.merge(service_df, on="USER_SEQ", how="left")
    df = df.merge(content_df, on="USER_SEQ", how="left")
    df = df.merge(broadcast_df, on="USER_SEQ", how="left")

    df["DEVICE_COUNT"] = df["DEVICE_COUNT"].fillna(0).astype(int)
    df["OTT_SERVICE_COUNT"] = df["OTT_SERVICE_COUNT"].fillna(0).astype(int)
    df["CONTENT_TYPE_COUNT"] = df["CONTENT_TYPE_COUNT"].fillna(0).astype(int)
    df["BROADCAST_TYPE_COUNT"] = df["BROADCAST_TYPE_COUNT"].fillna(0).astype(int)

    print("복수응답 집계 후 shape:", df.shape)

    # 5. 파생변수 생성
    df["AVG_MIN_WEEKDAY"] = pd.to_numeric(df["AVG_MIN_WEEKDAY"], errors="coerce").fillna(0)
    df["AVG_MIN_WEEKEND"] = pd.to_numeric(df["AVG_MIN_WEEKEND"], errors="coerce").fillna(0)
    df["SEARCH_VIEW"] = pd.to_numeric(df["SEARCH_VIEW"], errors="coerce").fillna(0)
    df["RECOMMEND_VIEW"] = pd.to_numeric(df["RECOMMEND_VIEW"], errors="coerce").fillna(0)
    df["BINGE_WATCH"] = pd.to_numeric(df["BINGE_WATCH"], errors="coerce").fillna(0)

    df["WEEKLY_TOTAL_MIN"] = df["AVG_MIN_WEEKDAY"] * 5 + df["AVG_MIN_WEEKEND"] * 2

    df["WEEKEND_RATIO"] = df["AVG_MIN_WEEKEND"] / (
        df["AVG_MIN_WEEKDAY"] + df["AVG_MIN_WEEKEND"] + 1
    )

    df["ENGAGEMENT_SCORE"] = (
        df["SEARCH_VIEW"] + df["RECOMMEND_VIEW"] + df["BINGE_WATCH"]
    )

    df["ACTIVE_DAYS_SCORE"] = (
        pd.to_numeric(df["USE_FREQUENCY"], errors="coerce").fillna(0) +
        (3 - pd.to_numeric(df["USED_LAST_WEEK"], errors="coerce").fillna(0))
    )

    print(df[[
        "USER_SEQ",
        "WEEKLY_TOTAL_MIN",
        "WEEKEND_RATIO",
        "ENGAGEMENT_SCORE",
        "ACTIVE_DAYS_SCORE",
        "DEVICE_COUNT",
        "OTT_SERVICE_COUNT",
        "CONTENT_TYPE_COUNT",
        "BROADCAST_TYPE_COUNT"
    ]].head())

    # 6. 위험도 라벨 생성
    def make_risk_label(row):
        score = 0

        use_frequency = pd.to_numeric(row["USE_FREQUENCY"], errors="coerce")
        used_last_week = pd.to_numeric(row["USED_LAST_WEEK"], errors="coerce")
        weekday_time = pd.to_numeric(row["WEEKDAY_TIME_CODE"], errors="coerce")
        weekend_time = pd.to_numeric(row["WEEKEND_TIME_CODE"], errors="coerce")
        monthly_fee = pd.to_numeric(row["MONTHLY_FEE_CODE"], errors="coerce")
        ad_intent = pd.to_numeric(row["AD_INTENT"], errors="coerce")

        if pd.isna(use_frequency):
            use_frequency = 0
        if pd.isna(used_last_week):
            used_last_week = 0
        if pd.isna(weekday_time):
            weekday_time = 0
        if pd.isna(weekend_time):
            weekend_time = 0
        if pd.isna(monthly_fee):
            monthly_fee = 0
        if pd.isna(ad_intent):
            ad_intent = 0

        # 최근 이용 여부
        if used_last_week == 2:
            score += 25

        # 이용 빈도: 적게 사용할수록 위험
        if use_frequency == 1:
            score += 25
        elif use_frequency == 2:
            score += 15
        elif use_frequency == 3:
            score += 5

        # 전체 시청 시간
        if row["WEEKLY_TOTAL_MIN"] < 180:
            score += 20
        elif row["WEEKLY_TOTAL_MIN"] < 300:
            score += 12
        elif row["WEEKLY_TOTAL_MIN"] < 450:
            score += 6

        # 참여도
        if row["ENGAGEMENT_SCORE"] <= 6:
            score += 18
        elif row["ENGAGEMENT_SCORE"] <= 9:
            score += 10
        elif row["ENGAGEMENT_SCORE"] <= 12:
            score += 4

        # 시청 시간 코드
        if weekday_time == 1:
            score += 8
        elif weekday_time == 2:
            score += 4

        if weekend_time == 1:
            score += 8
        elif weekend_time == 2:
            score += 4

        # 접속/이용 범위
        if row["DEVICE_COUNT"] <= 1:
            score += 8
        elif row["DEVICE_COUNT"] == 2:
            score += 4

        if row["OTT_SERVICE_COUNT"] <= 1:
            score += 8
        elif row["OTT_SERVICE_COUNT"] == 2:
            score += 4

        if row["CONTENT_TYPE_COUNT"] <= 1:
            score += 10
        elif row["CONTENT_TYPE_COUNT"] == 2:
            score += 6
        elif row["CONTENT_TYPE_COUNT"] == 3:
            score += 3

        if row["BROADCAST_TYPE_COUNT"] <= 1:
            score += 6
        elif row["BROADCAST_TYPE_COUNT"] == 2:
            score += 3

        # 요금/광고 관련
        if row["HAS_AD_PLAN"] in ["사용 중", "1", 1]:
            score += 4

        if monthly_fee == 1:
            score += 4

        if ad_intent == 1:
            score += 4

        if score >= 55:
            return "위험"
        elif score >= 30:
            return "주의"
        else:
            return "안전"

    df["RISK_GROUP"] = df.apply(make_risk_label, axis=1)

    print("\n라벨 분포:")
    print(df["RISK_GROUP"].value_counts())

    # 7. 학습용 입력 변수 선택
    # 파생변수도 실제 feature에 포함
    features = [
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
        "BROADCAST_TYPE_COUNT",
        "WEEKLY_TOTAL_MIN",
        "WEEKEND_RATIO",
        "ENGAGEMENT_SCORE",
        "ACTIVE_DAYS_SCORE"
    ]

    X = df[features].copy()
    y = df["RISK_GROUP"].copy()

    for col in X.columns:
        X[col] = X[col].fillna(0)

    categorical_cols = [
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
        "WEEKEND_TIME_CODE"
    ]

    for col in categorical_cols:
        X[col] = X[col].astype(str)

    numeric_cols = [
        "DEVICE_COUNT",
        "OTT_SERVICE_COUNT",
        "CONTENT_TYPE_COUNT",
        "BROADCAST_TYPE_COUNT",
        "WEEKLY_TOTAL_MIN",
        "WEEKEND_RATIO",
        "ENGAGEMENT_SCORE",
        "ACTIVE_DAYS_SCORE"
    ]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    cat_feature_indices = [X.columns.get_loc(col) for col in categorical_cols]

    print("\nCatBoost 입력 X shape:", X.shape)
    print("범주형 컬럼:", categorical_cols)

    # 8. 학습 / 테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # 9. 모델 학습
    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        verbose=50,
        auto_class_weights="Balanced",
        l2_leaf_reg=6,
        random_strength=1.5,
        min_data_in_leaf=20
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=80
    )

    # 10. 예측 및 성능 평가
    pred = model.predict(X_test)
    pred = pd.Series(pred.flatten())

    print("\n정확도:", accuracy_score(y_test, pred))
    print("\n혼동행렬:")
    print(confusion_matrix(y_test, pred, labels=["안전", "주의", "위험"]))

    print("\n분류 리포트:")
    print(classification_report(y_test, pred, digits=4))

    print("\n예측 라벨 분포:")
    print(pred.value_counts())

    # 11. 변수 중요도 확인
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    print("\n상위 중요 변수 20개:")
    print(importance_df.head(20))

    # 12. 모델 / 컬럼 / 범주형 컬럼 저장
    joblib.dump(model, "risk_model.pkl")
    joblib.dump(X.columns.tolist(), "model_columns.pkl")
    joblib.dump(categorical_cols, "categorical_columns.pkl")

    print("\n모델 저장 완료: risk_model.pkl")
    print("컬럼 저장 완료: model_columns.pkl")
    print("범주형 컬럼 저장 완료: categorical_columns.pkl")

except Exception as e:
    print("오류 발생:", e)

finally:
    if conn is not None:
        conn.close()
        print("DB 연결 종료")