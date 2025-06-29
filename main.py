from fastapi import FastAPI
from Script import run_bankruptcy_pipeline
app = FastAPI()
@app.get("/model")
def model_get():
    return run_bankruptcy_pipeline()


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
