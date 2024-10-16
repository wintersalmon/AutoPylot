from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class WebhookData(BaseModel):
    # Define your expected JSON structure here, e.g., `field_name: str`
    # For this example, let's keep it dynamic.
    data: dict

@app.post('/github/webhook')
async def github_webhook(request: Request):
    data = await request.json()
    # Process the webhook payload here
    print(f"Received data: {data}")
    return {"status": "success"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)