# main.py
import json
import os
import re
from fastapi import FastAPI, HTTPException, Request            # ■ 追加：FastAPI 関連
from pydantic import BaseModel                                 # ■ 追加：リクエストボディのバリデーション
from pyngrok import ngrok                                       # ■ 追加：ngrok
import boto3
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

# --- もとの Lambda から持ってきた関数群 ---

def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    return match.group(1) if match else "us-east-1"

# 環境変数からモデル ID を取得
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
bedrock_client = None

def init_bedrock_client(invoked_arn: str):
    global bedrock_client
    if bedrock_client is None:
        region = extract_region_from_arn(invoked_arn)
        bedrock_client = boto3.client('bedrock-runtime', region_name=region)
    return bedrock_client

def invoke_bedrock(payload: dict, context_arn: str):
    client = init_bedrock_client(context_arn)
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json"
    )
    # boto3 の StreamingBody を文字列に変換
    body_bytes = resp['body'].read() if isinstance(resp['body'], StreamingBody) else resp['body']
    return json.loads(body_bytes)

# --- FastAPI 用のモデル定義 ---

class ChatRequest(BaseModel):
    message: str
    conversationHistory: list = []

class ChatResponse(BaseModel):
    success: bool
    response: str
    conversationHistory: list

# --- FastAPI アプリケーション作成 ■ 追加／変更 ---

app = FastAPI(title="Gemma Chatbot API")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    """
    Lambda ハンドラ相当の処理を FastAPI エンドポイントに移植
    """
    try:
        # リクエストを Lambda イベント形式に組み替え
        lambda_event = {
            "body": json.dumps(req.dict()),
            "requestContext": {
                "authorizer": getattr(request.state, "authorizer", {})
            }
        }
        # Bedrock 呼び出し
        payload = json.loads(lambda_event["body"])
        bedrock_payload = {
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in payload["conversationHistory"] + [{"role":"user","content":payload["message"]}]
            ],
            "inferenceConfig": {
                "maxTokens": 512, "stopSequences": [], "temperature": 0.7, "topP": 0.9
            }
        }
        result = invoke_bedrock(bedrock_payload, context_arn=request.scope.get("aws.context", ""))
        # 応答メッセージ抽出
        msg = result['output']['message']['content'][0]['text']
        history = payload["conversationHistory"] + [{"role":"assistant","content":msg}]
        return {"success": True, "response": msg, "conversationHistory": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ngrok で外部公開 ■ 追加／変更 ---

if __name__ == "__main__":
    # ① ngrok トンネルを立ち上げて公開 URL を取得
    public_url = ngrok.connect(8000).public_url             # ■ 追加
    print("🚀 Public URL:", public_url)                     # ■ 追加

    # ② Uvicorn サーバーを起動
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
