import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 开发期直接代理到后端，避免 CORS 与 Range 头问题
      '/api': { target: 'http://127.0.0.1:8080', changeOrigin: true },
    },
  },
})
