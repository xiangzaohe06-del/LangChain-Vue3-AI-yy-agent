import axios, {type AxiosInstance, type AxiosRequestConfig, type AxiosResponse} from 'axios'


axios.defaults.headers['Content-Type'] = 'application/json;charset=utf-8'

// 创建axios实例
const service: AxiosInstance = axios.create({
    baseURL: '/dietitian/api',
    timeout: 10000
})

// request拦截器
service.interceptors.request.use(
    (config: AxiosRequestConfig & { headers: any }) => {
        return config
    },
    (error) => Promise.reject(error)
)

// 响应拦截器
service.interceptors.response.use(
    (response: AxiosResponse) => {
        return response.data
    },
    (error) => {
        return Promise.reject(error)
    }
)

// 封装 request 函数，支持泛型
const request = <T = any>(config: AxiosRequestConfig): Promise<T> => {
    // 注意这里指定返回类型 T
    return service.request<any, T>(config)
}


export default request
