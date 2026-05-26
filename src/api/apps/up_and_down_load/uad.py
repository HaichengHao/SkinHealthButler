# @Time    : 2026/5/26 21:56
# @Author  : hero
# @File    : uad.py
import os
from fastapi import APIRouter, UploadFile,HTTPException
from pathlib import Path
import aiofiles
from configs.project_default_configs import project_path
from loguru import logger
from uuid import uuid4



uad_route=APIRouter(
    prefix='/skin_imgs',
    tags=['skin_imgs'],

)


# @uad_route.post('/')
# async def uad():
#     return {
#         'statusCode': 200,
#         'message':'the image upload and download api has been invoked'
#     }


@uad_route.post('/upload/')
async def upload_img(file:UploadFile):
    suffix_allowed = ['.jpeg','.jpg']
    upload_dir = f'{project_path}/uploadimgs'
    os.makedirs(upload_dir, exist_ok=True)
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix in suffix_allowed:
        safe_filename = f'{uuid4().hex}{os.path.splitext(file.filename)[1]}'
        filepath = os.path.join(upload_dir,safe_filename)
        async with aiofiles.open(filepath, mode='wb') as f:
            while chunk:= await file.read(1024*1024):
                await f.write(chunk)

            return {
                'status': 'success',
                'code':200,
                'message': 'Uploaded successfully',
                'filename':file.filename,
                'safe_filename':safe_filename
            }


    else:
        raise HTTPException(
            status_code=404,
            detail='File type not allowed'
        )