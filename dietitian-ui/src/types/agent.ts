/**
 * 智能体对话请求参数接口
 */
export interface IAgentChat {
    // 文本消息
    message: string;
    //图片地址
    image_url?: string[];
    // 会话id
    thread_id: string;
    user_id?: string;
}
