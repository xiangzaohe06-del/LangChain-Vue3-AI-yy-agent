import os
import uuid

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from qcloud_cos import CosConfig, CosS3Client

load_dotenv()

router = APIRouter()


#   COS 配置
secret_id = os.getenv("COS_SECRET_ID")
secret_key = os.getenv("COS_SECRET_KEY")

app_id = os.getenv("COS_APP_ID")
region = os.getenv("COS_REGION") or "ap-beijing"

bucket = os.getenv("COS_BUCKET")

@router.get("/file/presign")
def get_presigned_url(filename: str):
    print(filename)
    """
    获取签证
    :param filename: 文件名
    :return: 签证实体
    """

    if not all((secret_id, secret_key, app_id, bucket)):
        raise HTTPException(status_code=503, detail="图片上传未配置，请填写 COS 环境变量")

    bucket_name = bucket if bucket.endswith(f"-{app_id}") else f"{bucket}-{app_id}"

    client = CosS3Client(CosConfig(
        Region=region,
        SecretId=secret_id,
        SecretKey=secret_key,
        Scheme="https"
    ))

    # 获取后缀
    ext = filename.split(".")[-1].lower() if "." in filename else "jpg"
    # 生成新的文件名
    uid = str(uuid.uuid4())
    new_filename = f"{uid}.{ext}"

    # 生成 COS 预签名上传 URL
    upload_url = client.get_presigned_url(
        Method="PUT",
        Bucket=bucket_name,
        Key=new_filename,
        Expired=120
    )

    # 生成访问地址
    access_url = f"https://{bucket_name}.cos.{region}.myqcloud.com/{new_filename}"

    return {
        "uploadUrl": upload_url,
        "accessUrl": access_url,
        "filename": new_filename,
    }
