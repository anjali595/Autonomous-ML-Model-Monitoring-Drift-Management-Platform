import Cookies from 'js-cookie';
import axios from 'axios';

const envAuthUrl = process.env.NEXT_PUBLIC_AUTH_URL;
const envApiUrl = process.env.NEXT_PUBLIC_API_URL;
let AUTH_BASE_URL = 'http://localhost:5000';

if (envAuthUrl) {
  AUTH_BASE_URL = envAuthUrl;
} else if (envApiUrl) {
  AUTH_BASE_URL = envApiUrl.replace(/\/api$/, '');
}

const authApi = axios.create({
  baseURL: AUTH_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach Authorization header from token cookie
authApi.interceptors.request.use(
  (config) => {
    const token = Cookies.get('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const login = async (email, password) => {
  const response = await authApi.post('/auth/login', { username: email, password });
  const { token } = response.data;
  Cookies.set('token', token);
  return response.data;
};

export const signup = async (email, password, name) => {
  const response = await authApi.post('/auth/register', { username: email, password });
  return response.data;
};

export const logout = () => {
  Cookies.remove('token');
  if (typeof window !== 'undefined') {
    window.location.href = '/login';
  }
};

export const getCurrentUser = async () => {
  const response = await authApi.get('/auth/me');
  return response.data;
};