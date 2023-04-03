from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

class Msg(BaseModel):
    msg: str

@app.post("/baseimg/{file.filename}")
async def create_upload_file(file: UploadFile = File(...), crop: str = Form(...)):
    return {"filename": file.filename, "crop": crop}

@app.route("/baseimg/{filename}", methods=["GET"])
async def check_upload_file(filename: str):
    return FileResponse(filename)



# ==============================================================================================================
# @app.get("/")
# async def root():
#     return {"message": "Hello World. Welcome to FastAPI!"}


# @app.get("/path")
# async def demo_get():
#     return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


# @app.post("/path")
# async def demo_post(inp: Msg):
#     return {"message": inp.msg.upper()}


# @app.get("/path/{path_id}")
# async def demo_get_path_id(path_id: int):
#     return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}