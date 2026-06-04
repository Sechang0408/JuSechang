"""
OTT 프로젝트 - CSV 데이터 Oracle DB 적재 스크립트
실행 전 준비: pip install oracledb pandas
"""

import pandas as pd
import oracledb

oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\app\oracle\product\11.2.0\server\bin")

# ============================================================
# Oracle 접속 정보 - 본인 환경에 맞게 수정
# ============================================================
DB_USER     = "scott"
DB_PASSWORD = "tiger"
DB_HOST     = "localhost"
DB_PORT     = 1521
DB_SERVICE  = "xe"


# ============================================================
# 1. CSV 로드 및 전처리
# ============================================================
df = pd.read_csv("오티티_1차프로젝트.csv", encoding='cp949')

# 유료 OTT 미이용자(코드 98) 제거
df = df[df[df.columns[0]] != 98].reset_index(drop=True)

# 가구소득 무응답(9) 제거
df = df[df['가구소득'] != 9].reset_index(drop=True)
cols = df.columns.tolist()

# 단일 응답 컬럼 rename
df = df.rename(columns={
    cols[22]: 'MONTHLY_FEE_CODE',
    cols[23]: 'HAS_AD_PLAN',
    cols[24]: 'AD_CONTINUE_INTENT_RAW',
    cols[26]: 'AD_FUTURE_INTENT_RAW',
    cols[27]: 'USE_FREQUENCY',
    cols[33]: 'WEEKDAY_TIME_CODE',
    cols[34]: 'WEEKEND_TIME_CODE',
    cols[35]: 'USED_LAST_WEEK',
    cols[36]: 'AVG_MIN_WEEKDAY',
    cols[37]: 'AVG_MIN_WEEKEND',
    cols[50]: 'SEARCH_VIEW',
    cols[51]: 'RECOMMEND_VIEW',
    cols[52]: 'BINGE_WATCH',
    '성별':   'GENDER',
    '연령':   'AGE_GROUP',
    '가구소득': 'INCOME_GROUP',
    '가족구성': 'FAMILY_TYPE',
})

# 복수응답 컬럼 rename
b14_cols  = [cols[i] for i in range(0, 22)]
b18_cols  = [cols[i] for i in range(28, 33)]
b23_cols  = [cols[i] for i in range(38, 44)]
b231_cols = [cols[i] for i in range(44, 50)]

df = df.rename(columns={
    **{c: f'OTT_SERVICE_{i+1}'    for i, c in enumerate(b14_cols)},
    **{c: f'DEVICE_{i+1}'         for i, c in enumerate(b18_cols)},
    **{c: f'CONTENT_TYPE_{i+1}'   for i, c in enumerate(b23_cols)},
    **{c: f'BROADCAST_TYPE_{i+1}' for i, c in enumerate(b231_cols)},
})

# AD_INTENT 통합 컬럼 생성
def build_ad_intent(row):
    if pd.isna(row['HAS_AD_PLAN']):
        return None
    elif int(row['HAS_AD_PLAN']) == 1:
        v = row['AD_CONTINUE_INTENT_RAW']
        return int(v) if pd.notna(v) else None
    else:
        v = row['AD_FUTURE_INTENT_RAW']
        return int(v) if pd.notna(v) else None

df['AD_INTENT'] = df.apply(build_ad_intent, axis=1)
df = df[df['AD_INTENT'].notna()].reset_index(drop=True)

def to_int_or_none(val):
    return int(val) if pd.notna(val) else None

def to_float_or_none(val):
    return float(val) if pd.notna(val) else None


# ============================================================
# 2. Oracle 접속
# ============================================================
conn   = oracledb.connect(user=DB_USER, password=DB_PASSWORD,
                          host=DB_HOST, port=DB_PORT, service_name=DB_SERVICE)
cursor = conn.cursor()
print("Oracle 접속 완료")

try:
    for _, row in df.iterrows():

        # ── USERS INSERT + USER_SEQ 반환 ────────────────────
        # RETURNING ~ INTO 로 Oracle 시퀀스가 생성한 USER_SEQ를 바로 받아옴
        user_seq_var = cursor.var(int)

        cursor.execute(
            """INSERT INTO USERS (USER_SEQ, GENDER, AGE_GROUP)
               VALUES (SEQ_USER.NEXTVAL, :1, :2)
               RETURNING USER_SEQ INTO :3""",
            [int(row['GENDER']), int(row['AGE_GROUP']), user_seq_var]
        )
        user_seq = user_seq_var.getvalue()[0]   # 방금 생성된 USER_SEQ

        # ── ECONOMY INSERT ───────────────────────────────────
        cursor.execute(
            """INSERT INTO ECONOMY (USER_SEQ, INCOME_GROUP, FAMILY_TYPE)
               VALUES (:1, :2, :3)""",
            [user_seq, int(row['INCOME_GROUP']), int(row['FAMILY_TYPE'])]
        )

        # ── SUBSCRIPTION INSERT ──────────────────────────────
        cursor.execute(
            """INSERT INTO SUBSCRIPTION
                   (USER_SEQ, MONTHLY_FEE_CODE, HAS_AD_PLAN, AD_INTENT)
               VALUES (:1, :2, :3, :4)""",
            [
                user_seq,
                to_int_or_none(row['MONTHLY_FEE_CODE']),
                to_int_or_none(row['HAS_AD_PLAN']),
                row['AD_INTENT'],
            ]
        )

        # ── USAGE_BEHAVIOR INSERT ────────────────────────────
        cursor.execute(
            """INSERT INTO USAGE_BEHAVIOR
                   (USER_SEQ, USE_FREQUENCY, WEEKDAY_TIME_CODE, WEEKEND_TIME_CODE,
                    USED_LAST_WEEK, AVG_MIN_WEEKDAY, AVG_MIN_WEEKEND,
                    SEARCH_VIEW, RECOMMEND_VIEW, BINGE_WATCH)
               VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)""",
            [
                user_seq,
                to_int_or_none(row['USE_FREQUENCY']),
                to_int_or_none(row['WEEKDAY_TIME_CODE']),
                to_int_or_none(row['WEEKEND_TIME_CODE']),
                to_int_or_none(row['USED_LAST_WEEK']),
                to_float_or_none(row['AVG_MIN_WEEKDAY']),
                to_float_or_none(row['AVG_MIN_WEEKEND']),
                to_int_or_none(row['SEARCH_VIEW']),
                to_int_or_none(row['RECOMMEND_VIEW']),
                to_int_or_none(row['BINGE_WATCH']),
            ]
        )

        # ── 복수응답 테이블 INSERT ───────────────────────────
        multi_tables = [
            ("USER_OTT_SERVICE",    "SERVICE_CODE",   [f'OTT_SERVICE_{i}'    for i in range(1, 23)]),
            ("USER_DEVICE",         "DEVICE_CODE",    [f'DEVICE_{i}'         for i in range(1, 6)]),
            ("USER_CONTENT_TYPE",   "CONTENT_CODE",   [f'CONTENT_TYPE_{i}'   for i in range(1, 7)]),
            ("USER_BROADCAST_TYPE", "BROADCAST_CODE", [f'BROADCAST_TYPE_{i}' for i in range(1, 7)]),
        ]

        for table, code_col, response_cols in multi_tables:
            seen = set()
            for c in response_cols:
                v = row[c]
                if pd.notna(v):
                    code = int(v)
                    if code not in seen:
                        seen.add(code)
                        cursor.execute(
                            f"INSERT INTO {table} (USER_SEQ, {code_col}) VALUES (:1, :2)",
                            [user_seq, code]
                        )

    conn.commit()
    print(f"커밋 완료 - {len(df)}명 전체 적재 성공")

except Exception as e:
    conn.rollback()
    print(f"[ERROR] 롤백 처리됨: {e}")
    raise

finally:
    cursor.close()
    conn.close()
    print("Oracle 접속 종료")
