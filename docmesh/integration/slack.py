import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # Socket Mode용 앱 토큰 (xapp-...)
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))

# Slack Bolt 앱 초기화
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)


@app.command("/docqa")
def handle_docqa_command(ack, command, respond):
    # 슬래시 커맨드 수신 시 즉시 응답(ack)하여 Slack 타임아웃 방지
    ack()
    query = command.get("text", "").strip()
    if not query:
        respond("질문을 입력해 주세요.")
        return

    try:
        # /ask 엔드포인트에 질문 전달
        api_response = requests.post(
            f"{FASTAPI_HOST}:{FASTAPI_PORT}", json={"question": query}
        )
        if api_response.status_code != 200:
            respond("답변 생성에 실패했습니다.")
            return
        data = api_response.json()
        answer = data.get("answer", "답변을 가져올 수 없습니다.")
        # Slack에 답변 전송
        respond(answer)
    except Exception as e:
        print(f"Error: {e}")
        respond("답변 생성 중 오류가 발생했습니다.")


if __name__ == "__main__":
    # Slack Socket Mode 핸들러 실행 (슬랙 앱 설정에서 Socket Mode 활성화 필요)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
