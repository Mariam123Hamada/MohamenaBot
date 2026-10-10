import uvicorn


def main() -> None:
    uvicorn.run("mohamenabot.api:app", host="0.0.0.0", port=8000, reload=False)
