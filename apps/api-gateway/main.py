from fastapi import FastAPI


app = FastAPI(title="CosmosAI API Gateway")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway"}
