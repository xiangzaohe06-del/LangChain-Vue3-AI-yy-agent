import request from "../utils/request.ts";
import type {IPresign} from "@/types/file.ts";

/**
 * 获取预签名
 * @param filename
 */
export function getPresignedUrl(filename: string) {
    return request({
        url: '/file/presign',
        method: 'get',
        params: {
            filename
        }
    })
}

/**
 * 上传文件
 * @param url
 * @param file
 */
export function uploadCosFile(url: string, file: File) {
    return request<IPresign>({
        url: url,
        method: 'PUT',
        data: file
    })
}
