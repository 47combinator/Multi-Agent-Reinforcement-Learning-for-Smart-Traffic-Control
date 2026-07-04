@echo off
echo ============================================================
echo   Git Push Script - DQN Traffic Signal Control Project
echo ============================================================
echo.

echo [1/5] Initializing Git repository...
git init

echo [2/5] Setting up Git remote origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/47combinator/Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control.git

echo [3/5] Creating and switching to branch 'feature-dqn-agent'...
git checkout -b feature-dqn-agent >nul 2>&1
if errorlevel 1 (
    git checkout feature-dqn-agent
)

echo [4/5] Staging files...
git add .

echo [5/5] Committing code...
git commit -m "Implement Deep Q-Network (DQN) agent with Double Dueling DQN, PER, baselines (PPO, Q-Learning), and evaluations"

echo.
echo ============================================================
echo   Pushing code to GitHub...
echo ============================================================
git push -u origin feature-dqn-agent

echo.
echo Operation completed!
pause
