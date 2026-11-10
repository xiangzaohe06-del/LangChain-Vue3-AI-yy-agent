/**
 * 聊天消息对话展示接口
 */
export interface IChatMessage {
    role: "user" | "assistant",
    message: string,
    image_url?: string[]
}
