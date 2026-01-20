# HyperQuant AI - 期权量化交易平台

<div align="center">

![HyperQuant AI](https://img.shields.io/badge/HyperQuant-AI%20Quant%20Platform-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-green?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-16+-black?style=flat-square&logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=flat-square&logo=fastapi)

**一个现代化的期权量化交易研究平台，支持策略开发、回测分析和 AI 辅助决策**

</div>

---

## ✨ 功能特性

### 🚀 策略工坊 (Foundry)
- **代码编辑器** - Monaco Editor 驱动的策略开发环境
- **策略管理** - 创建、编辑、保存策略文件
- **AI Copilot** - 智能代码助手，辅助策略编写
- **数据管理** - 支持平台数据和用户自定义数据

### 📊 驾驶舱 (Cockpit)
- **回测引擎** - 真实历史数据驱动的策略回测
- **权益曲线** - 实时展示回测结果和关键指标
- **期权链** - T 型报价和 Greeks 可视化
- **波动率曲面** - 3D 隐含波动率可视化
- **波动率锥** - 历史波动率分析
- **蒙特卡洛模拟** - 多种随机过程模型 (GBM, Heston, MJD, GARCH)
- **希腊值计算器** - 实时计算 Delta, Gamma, Theta, Vega, Rho
- **保证金分析** - 风险敞口监控

### 🤖 AI 功能
- **策略建议** - 基于市场状态的智能建议
- **AI 分析报告** - 自动生成回测分析报告
- **实时提示** - 交互式 AI 提示栏

---

## 📁 项目结构

```
Option-sim/
├── frontend/                # Next.js 前端
│   ├── app/                 # 页面路由
│   │   ├── page.tsx         # 策略工坊页面
│   │   └── cockpit/         # 驾驶舱页面
│   └── components/          # React 组件
│       ├── foundry/         # 策略工坊组件
│       │   ├── CodeEditor.tsx
│       │   ├── FileExplorer.tsx
│       │   ├── AICopilot.tsx
│       │   └── ModelDataManager.tsx
│       └── cockpit/         # 驾驶舱组件
│           ├── MissionControl.tsx
│           ├── HolographicDashboard.tsx
│           └── charts/      # 图表组件
│
├── backend/                 # FastAPI 后端
│   ├── main.py              # 应用入口
│   ├── requirements.txt     # Python 依赖
│   └── app/
│       ├── api/             # API 路由
│       │   ├── backtest_api.py
│       │   ├── data.py
│       │   ├── strategies.py
│       │   ├── greeks.py
│       │   ├── simulation.py
│       │   └── files.py
│       ├── engines/         # 核心引擎
│       └── models/          # 数据模型
│
├── data/                    # 期权历史数据 (Parquet)
│   └── 510050_SH/           # 50ETF 期权数据
│       ├── 2019/
│       ├── 2020/
│       └── ...
│
├── strategies/              # 策略模板
│
└── start_platform.bat       # 一键启动脚本 (Windows)
```

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| **Node.js** | 18.0+ | 前端运行环境 |
| **Python** | 3.9+ | 后端运行环境 |
| **npm** | 8.0+ | 包管理器 |

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/Option-sim.git
cd Option-sim
```

#### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

#### 3. 安装前端依赖

```bash
cd ../frontend
npm install
```

#### 4. 启动服务

**方式一：一键启动 (Windows)**
```bash
cd ..
start_platform.bat
```

**方式二：手动启动**

```bash
# 终端 1: 启动后端
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 终端 2: 启动前端
cd frontend
npm run dev
```

#### 5. 访问平台

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| 后端 API 文档 | http://localhost:8000/docs |

---

## 💡 使用指南

### 策略开发流程

1. **打开策略工坊** - 访问 http://localhost:3000
2. **创建新策略** - 右键点击文件列表，选择"New File"
3. **编写策略代码** - 使用 Monaco 编辑器编写 Python 策略
4. **保存策略** - Ctrl+S 保存

### 回测流程

1. **切换到驾驶舱** - 点击导航栏进入 `/cockpit`
2. **配置回测参数** - 在左侧 Mission Control 面板设置：
   - 选择策略
   - 选择数据集
   - 设置时间范围
   - 配置保证金和风控参数
3. **运行回测** - 点击 "RUN" 按钮
4. **分析结果** - 查看权益曲线、交易记录、绩效指标

---

## 📊 数据说明

项目包含 **50ETF 期权历史数据**，格式为 Parquet，位于 `data/510050_SH/` 目录：

```
data/510050_SH/
├── 2019/
│   ├── 202001.parquet
│   ├── 202002.parquet
│   └── ...
├── 2020/
└── ...
```

每个文件包含该月所有期权合约的行情数据，字段包括：
- 合约代码、行权价、到期日
- 开盘价、最高价、最低价、收盘价
- 隐含波动率 (IV)
- Greeks (Delta, Gamma, Theta, Vega)

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **前端框架** | Next.js 16, React 19 |
| **UI 库** | Tailwind CSS, Framer Motion |
| **图表** | Plotly.js, Recharts |
| **代码编辑器** | Monaco Editor |
| **后端框架** | FastAPI |
| **数据处理** | Pandas, NumPy, Numba |
| **科学计算** | SciPy |

---

## 🤝 开发指南

### 添加新策略

策略文件放在 `backend/strategies_storage/` 目录，遵循以下模板：

```python
class MyStrategy:
    """策略描述"""
    
    def __init__(self, config):
        self.config = config
    
    def on_bar(self, bar_data):
        """每根 K 线触发"""
        pass
    
    def on_order_filled(self, order):
        """订单成交回调"""
        pass
```

### 添加新 API

1. 在 `backend/app/api/` 创建新路由文件
2. 在 `backend/main.py` 注册路由

---

## 📝 常见问题

**Q: 前端启动报错 "Module not found"**

运行 `npm install` 重新安装依赖。

**Q: 后端启动报错 "No module named ..."**

运行 `pip install -r requirements.txt` 安装依赖。

**Q: 回测没有数据**

确认 `data/510050_SH/` 目录包含 Parquet 数据文件。

---

## 📄 License

MIT License

---

<div align="center">

**Made with ❤️ for Quantitative Trading**

</div>
