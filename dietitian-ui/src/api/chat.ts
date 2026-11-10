import type {IAgentChat} from "@/types/agent.ts";
import axios from "axios";
import request from "@/utils/request.ts";
import {userHeaders} from '@/utils/user';

/**
 * 使用 Axios 接收流式对话
 * @param params 请求体
 * @param onChunk 收到文本碎片的的回调
 * @param onDone 完成后的回调
 * @param onError 错误回调
 */
export async function chatStream(
    params: IAgentChat,
    onChunk: (text: string) => void,
    onDone?: () => void,
    onError?: (err: any) => void
) {
    try {
        const response = await axios.post('/dietitian/api/chat/stream', params, {
            headers: {
                'Accept': 'text/event-stream',
            },
            // 强制 Axios 使用 fetch 适配器以完美支持浏览器端的 ReadableStream
            adapter: 'fetch',
            responseType: 'stream'
        });

        // 此时 response.data 是一个原生的 ReadableStream
        const stream = response.data as ReadableStream;
        const reader = stream.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunkText = decoder.decode(value, { stream: true });
            onChunk(chunkText);
        }

        if (onDone) onDone();
    } catch (error) {
        if (onError) onError(error);
        else console.error('Axios stream error:', error);
    }
}


/**
 * 获取历史记录
 * @param threadId
 */
export function getChatHistory(threadId: string) {
    return request({
        url: '/chat/history/'+threadId,
        method: 'get',
        headers: userHeaders(),
    })
}



/**
 * 清空聊天记录
 * @param threadId
 */
export function clearChatHistory(threadId: string) {
    return request({
        url: '/chat/history/'+threadId,
        method: 'delete',
        headers: userHeaders(),
    })
}

