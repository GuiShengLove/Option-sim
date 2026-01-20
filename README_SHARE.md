# 项目分享指南

## 如何分享给组员

### 方法1: 压缩整个项目文件夹 (推荐)

1. **删除不必要的大文件夹**（节省空间）：
   - `frontend/node_modules/` - 可以删除，组员自己安装
   - `backend/__pycache__/` - 可以删除

2. **压缩** `Option-sim` 文件夹

3. **组员收到后**执行：
   ```bash
   # 安装前端依赖
   cd frontend
   npm install
   
   # 安装后端依赖  
   cd ../backend
   pip install -r requirements.txt
   ```

### 方法2: 使用 Git (最佳实践)

1. **初始化Git仓库**：
   ```bash
   cd e:\Quant_code\Option-sim
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **推送到GitHub/GitLab**：
   ```bash
   git remote add origin https://github.com/your-repo.git
   git push -u origin main
   ```

3. **组员克隆**：
   ```bash
   git clone https://github.com/your-repo.git
   cd Option-sim
   ```

---

## 组员环境配置

### 必要软件
- **Node.js** 18+ (前端)
- **Python** 3.9+ (后端)

### 启动命令

**1. 启动后端**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**2. 启动前端**
```bash
cd frontend
npm install
npm run dev
```

**3. 访问**
- 前端: http://localhost:3000
- 后端API: http://localhost:8000

---

## 重要文件说明

| 路径 | 说明 |
|------|------|
| `data/510050_SH/` | 期权历史数据 (Parquet格式) |
| `backend/.env` | 后端配置 (API密钥等) |
| `frontend/components/` | React组件 |
| `backend/app/api/` | FastAPI路由 |
