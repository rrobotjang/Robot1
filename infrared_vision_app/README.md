# Scratch Infrared Vision App

적외선 카메라 기반 안전관제 서비스를 `스크래치부터 다시 시작한` 앱이며, 이제는 게이트웨이 중심 구조에서 `외부 IAM`, `실시간 메시지 큐`, `CV 추론 서비스`, `워커`, `다중 인스턴스 배포`까지 분리할 수 있도록 확장했다.

## 포함 구성

- FastAPI 기반 백엔드 API
- 정적 프런트엔드 대시보드
- 물류센터 / 도로 시나리오용 목업 데이터
- 게이트웨이 기반 추론 API와 자동 이벤트 생성
- 슬라이딩 윈도우 기반 열화상 스캔 추론
- 벡터 검색 기반 대응 가이드 조회
- 외부 IAM 연동 가능 인증 게이트웨이
- 액세스 토큰 + 리프레시 토큰 기반 세션 수명 관리
- SQLite 기반 이벤트 영속 저장
- SQLite 기반 세션 저장과 감사 로그
- 시스템 상태 및 감사 로그 API
- 표준화된 요청 ID / 에러 응답
- 모바일 운영 화면 자동 동기화
- Redis 기반 실시간 추론 잡 큐
- 독립 실행 가능한 CV 서비스 앱
- 독립 실행 가능한 IAM 서비스 앱
- 추론 워커 프로세스
- Nginx 기반 다중 인스턴스 라우팅 예시
- Docker 실행 예시

## 분산 아키텍처

- `app.py`: 운영자 UI와 도메인 API를 제공하는 게이트웨이
- `iam_service_app.py`: 외부 IAM처럼 독립 배포 가능한 인증 서비스
- `cv_service_app.py`: 실제 CV 추론을 맡는 전용 서비스
- `inference_worker.py`: 메시지 큐에서 잡을 가져와 추론을 수행하는 워커
- `job_queue.py`: 메모리/Redis 기반 잡 큐 어댑터
- `docker-compose.distributed.yml`: `gateway_a`, `gateway_b`, `nginx`, `redis`, `infrared_iam`, `infrared_cv_service`, `infrared_worker` 구성
- `docker-compose.edge.yml`: FSD edge computer용 단일 노드 구성

## 별론: FSD Edge Computer 관점

- 현재의 분산 아키텍처는 `중앙 관제 또는 서버형 배포`를 기준으로 정리한 것이다.
- FSD에 실제로 사용 중인 `edge computer`를 기준으로 보면, 주 실행 노드는 차량 내부의 단일 컴퓨팅 장치가 된다.
- 이 경우 권장 기본 형태는 `single gateway + local CV + local queue + local store`이며, `gateway_a/gateway_b` 같은 다중 게이트웨이는 기본 구성이 아니라 선택적 확장 요소다.
- 엣지 컴퓨팅 모드에서는 `DB_BACKEND=memory` 또는 `sqlite`, `QUEUE_BACKEND=memory`, `IAM_MODE=local`, `CV_SERVICE_MODE=local`이 자연스럽다.
- 중앙 PostgreSQL, Redis, 외부 IAM은 실시간 판단 경로가 아니라 `후행 동기화`, `통합 관제`, `운영 로그 수집` 용도로 붙이는 것이 더 적합하다.
- 즉 `FSD edge computer가 1차 판단과 제어의 중심`, 중앙 인프라는 `보조 분석과 운영 관리` 역할로 분리하는 것이 권장 방향이다.

## Edge 실행 예시

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
cp .env.edge.example .env.edge
docker compose -f docker-compose.edge.yml up --build
```

- 기본값은 `single gateway + local CV + memory queue + sqlite local store`다.
- 더 가볍게 쓰려면 `.env.edge`에서 `DB_BACKEND=memory`로 바꾸면 된다.
- 브라우저 진입점은 `http://127.0.0.1:8010`이다.

## 실행 방법

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8010
```

브라우저에서 `http://127.0.0.1:8010`으로 접속하면 된다.

## API 예시

- `POST /api/auth/login`
- `POST /api/auth/refresh`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET /api/health`
- `GET /api/system/status`
- `GET /api/dashboard?profile=logistics`
- `GET /api/events?profile=road`
- `GET /api/events/{event_id}`
- `POST /api/events/{event_id}/acknowledge`
- `POST /api/events/{event_id}/status`
- `POST /api/inference/run`
- `POST /api/inference/jobs`
- `GET /api/inference/jobs/{job_id}`
- `GET /api/vector/status`
- `POST /api/vector/search`
- `GET /api/audit/logs`

## 데모 계정

- `ops_admin / demo123!`
- `field_operator / demo123!`
- `road_manager / demo123!`
- `safety_viewer / demo123!`

## 환경 변수

- `APP_NAME`: 서비스 이름
- `APP_VERSION`: 앱 버전
- `APP_RELEASE_CHANNEL`: 배포 채널 표시값
- `AUTH_SECRET`: 인증 토큰 서명 키
- `AUTH_EXPIRES_MINUTES`: 로그인 유지 시간
- `AUTH_REFRESH_EXPIRES_DAYS`: 리프레시 토큰 유지 일수
- `VECTOR_DB_BACKEND`: `memory` 또는 `qdrant`
- `QDRANT_URL`: Qdrant 서버 주소
- `QDRANT_COLLECTION`: Qdrant 컬렉션 이름
- `QDRANT_TIMEOUT`: Qdrant 요청 타임아웃(초)
- `DB_BACKEND`: `memory`, `sqlite` 또는 `postgresql`
- `DB_URL`: PostgreSQL 연결 문자열
- `APP_DB_PATH`: SQLite 데이터베이스 파일 경로
- `CORS_ALLOW_ORIGINS`: 허용할 Origin 목록
- `AUTO_REFRESH_SECONDS`: 운영 화면 자동 동기화 주기
- `IAM_MODE`: `local` 또는 `external`
- `IAM_SERVICE_URL`: 외부 IAM 서비스 주소
- `CV_SERVICE_MODE`: `local` 또는 `remote`
- `CV_SERVICE_URL`: 외부 CV 서비스 주소
- `CV_SERVICE_TIMEOUT`: 외부 CV 서비스 타임아웃
- `QUEUE_BACKEND`: `memory` 또는 `redis`
- `QUEUE_URL`: Redis 연결 문자열
- `QUEUE_NAME`: 추론 잡 큐 이름
- `QUEUE_RESULT_TTL_SECONDS`: 잡 결과 보관 시간(초)

## Docker 실행 예시

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
docker compose up --build
```

## 분산 실행 예시

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
cp .env.distributed.example .env.distributed
docker compose -f docker-compose.distributed.yml up --build
```

브라우저 진입점은 `http://127.0.0.1:8080`이다.

## PostgreSQL 마이그레이션 명령

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
/Users/robotjang/Documents/Robot2/infrared_vision_app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py \
  --sqlite-path /Users/robotjang/Documents/Robot2/infrared_vision_app/data/infrared_vision.db \
  --postgres-dsn postgresql://infrared:infrared@127.0.0.1:5432/infrared_vision \
  --truncate
```

분산 스택을 먼저 올린 뒤 마이그레이션을 수행하려면:

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
cp .env.distributed.example .env.distributed
docker compose -f docker-compose.distributed.yml up -d postgres redis
/Users/robotjang/Documents/Robot2/infrared_vision_app/.venv/bin/python scripts/migrate_sqlite_to_postgres.py \
  --sqlite-path /Users/robotjang/Documents/Robot2/infrared_vision_app/data/infrared_vision.db \
  --postgres-dsn postgresql://infrared:infrared@127.0.0.1:5432/infrared_vision \
  --truncate
docker compose -f docker-compose.distributed.yml up -d --build
```

## 개발 테스트

```bash
cd /Users/robotjang/Documents/Robot2/infrared_vision_app
python -m pip install -r requirements-dev.txt
pytest
```

## 슬라이딩 윈도우 입력 예시

`POST /api/inference/run` 호출 시 아래 필드를 함께 보내면 느리지만 직관적인 열화상 스캔 방식을 사용할 수 있다.

```json
{
  "profile": "logistics",
  "camera_id": "CAM-L-01",
  "thermal_summary": "하역구역 일부 구간에서 국부 고온 영역이 보임",
  "use_sliding_window": true,
  "thermal_matrix": [
    [0.12, 0.15, 0.20, 0.18],
    [0.18, 0.82, 0.88, 0.24],
    [0.20, 0.84, 0.91, 0.26],
    [0.12, 0.18, 0.22, 0.15]
  ]
}
```

응답에는 `detection_strategy`, `scanned_windows`, `hotspot_count`가 포함되어 슬라이딩 윈도우 스캔 결과를 바로 확인할 수 있다.

## PostgreSQL 전환

- 기본 테스트와 단일 실행은 그대로 SQLite를 사용할 수 있다.
- 엣지 실행은 `.env.edge.example` 기준으로 `memory` 또는 `sqlite`를 선택할 수 있다.
- 분산 배포는 [docker-compose.distributed.yml](/Users/robotjang/Documents/Robot2/infrared_vision_app/docker-compose.distributed.yml)에 맞춰 PostgreSQL을 중앙 DB로 사용한다.
- 저장소 팩토리는 `DB_BACKEND=postgresql`과 `DB_URL`이 주어지면 PostgreSQL store를 사용하고, 아니면 SQLite store를 사용한다.

## 현재 범위

- 적외선 카메라 영상은 실제 스트림 대신 운영 화면용 목업 피드로 표시한다.
- 현재 CV 서비스는 규칙 기반 + 슬라이딩 윈도우 스캔 로직을 사용하며, 딥러닝 모델 서버로 교체 가능한 HTTP 계약을 갖는다.
- 현재 스캐폴드는 `설정 분리`, `세션 회전`, `로그아웃`, `감사 로그`, `모바일 자동 동기화`, `외부 IAM`, `Redis 큐`, `CV 서비스`, `워커`, `기본 테스트`까지 포함한다.
- 다중 인스턴스 예시는 이제 PostgreSQL 기반으로 정리되어 있고, 엣지 단일 노드는 `memory` 또는 `sqlite`를 선택하는 구성을 권장한다.
