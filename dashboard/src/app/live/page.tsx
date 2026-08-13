"use client";

import React, { useEffect, useState, useMemo } from "react";
import io, { Socket } from "socket.io-client";
import { Play, Square, Pause, Activity, Clock, BarChart3, Download } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area, Legend } from "recharts";
import { Button, Card, Stat } from "@/components/ui";
import { cn } from "@/lib/utils";
import { exportToCSV, exportToJSON } from "@/lib/export";

type Metric = {
  episode: number;
  step: number;
  reward: number;
  wait_time: number;
  queue_length: number;
  throughput: number;
  action: number;
  delay?: number;
};

export default function LiveSimulation() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  
  // Settings
  const [model, setModel] = useState("PPO");
  const [episodes, setEpisodes] = useState(1);
  const [seed, setSeed] = useState(42);
  const [useGui, setUseGui] = useState(false);
  
  // Status
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  
  // Data
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [currentMetric, setCurrentMetric] = useState<Metric | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState("00:00");

  useEffect(() => {
    const newSocket = io("http://localhost:5000", {
      reconnectionAttempts: 5,
      timeout: 10000,
    });
    setSocket(newSocket);

    newSocket.on("connect", () => {
      setConnected(true);
      fetch("http://localhost:5000/api/status")
        .then(res => res.json())
        .then(data => {
          setIsRunning(data.running);
          setIsPaused(data.paused);
        });
    });
    newSocket.on("disconnect", () => setConnected(false));
    
    newSocket.on("metrics", (data: Metric) => {
      setCurrentMetric(data);
      setMetrics((prev) => {
        const updated = [...prev, data];
        if (updated.length > 1000) return updated.slice(updated.length - 1000);
        return updated;
      });
    });

    newSocket.on("simulation_complete", () => {
      setIsRunning(false);
      setIsPaused(false);
    });

    newSocket.on("error", (err) => {
      console.error(err);
      setIsRunning(false);
      setIsPaused(false);
      alert(err.message || "An error occurred");
    });

    return () => { newSocket.close(); };
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRunning && !isPaused && startTime) {
      interval = setInterval(() => {
        const diff = Math.floor((Date.now() - startTime) / 1000);
        const m = Math.floor(diff / 60).toString().padStart(2, '0');
        const s = (diff % 60).toString().padStart(2, '0');
        setElapsed(`${m}:${s}`);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRunning, isPaused, startTime]);

  const handleStart = () => {
    if (!socket) return;
    setMetrics([]);
    setCurrentMetric(null);
    setIsRunning(true);
    setIsPaused(false);
    setStartTime(Date.now());
    socket.emit("start_simulation", { model, episodes, seed, use_gui: useGui });
  };

  const handleStop = async () => {
    await fetch("http://localhost:5000/api/stop", { method: "POST" });
  };

  const handlePause = async () => {
    await fetch("http://localhost:5000/api/pause", { method: "POST" });
    setIsPaused(true);
  };

  const handleResume = async () => {
    await fetch("http://localhost:5000/api/resume", { method: "POST" });
    setIsPaused(false);
  };

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto w-full">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white">Live Simulation</h2>
          <p className="text-[#8890a8] mt-1 text-sm flex items-center gap-2">
            Status: 
            <span className={cn("px-2 py-0.5 rounded-full text-[10px] font-bold uppercase", 
              connected ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
            )}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" onClick={() => exportToCSV(metrics, "live_simulation")} disabled={metrics.length === 0}>
            <Download size={16} /> CSV
          </Button>
          <Button variant="outline" onClick={() => exportToJSON(metrics, "live_simulation")} disabled={metrics.length === 0}>
            <Download size={16} /> JSON
          </Button>
        </div>
      </header>

      {/* Control Panel */}
      <Card className="flex flex-wrap gap-6 items-end border-blue-500/20 bg-blue-500/5">
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-[#8890a8] uppercase tracking-wider">Model</label>
          <select 
            value={model} 
            onChange={(e) => setModel(e.target.value)}
            disabled={isRunning}
            className="bg-[#12141c] border border-[#3a3f55]/50 text-sm rounded-lg px-3 py-2 text-white outline-none focus:border-blue-500"
          >
            <option>PPO</option>
            <option>DQN</option>
            <option>Q-Learning</option>
            <option>Fixed-Time</option>
          </select>
        </div>
        
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-[#8890a8] uppercase tracking-wider">Episodes</label>
          <input 
            type="number" value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} disabled={isRunning}
            className="bg-[#12141c] border border-[#3a3f55]/50 text-sm rounded-lg px-3 py-2 text-white outline-none focus:border-blue-500 w-24"
          />
        </div>
        
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-[#8890a8] uppercase tracking-wider">Seed</label>
          <input 
            type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} disabled={isRunning}
            className="bg-[#12141c] border border-[#3a3f55]/50 text-sm rounded-lg px-3 py-2 text-white outline-none focus:border-blue-500 w-24"
          />
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold text-[#8890a8] uppercase tracking-wider">GUI</label>
          <button
            onClick={() => setUseGui(!useGui)}
            disabled={isRunning}
            className={cn("w-12 h-6 rounded-full relative transition-colors", useGui ? "bg-blue-500" : "bg-[#3a3f55]")}
          >
            <div className={cn("absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform", useGui ? "translate-x-6" : "translate-x-0")} />
          </button>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-3">
          {!isRunning ? (
            <Button onClick={handleStart} disabled={!connected}><Play size={16} /> Start</Button>
          ) : (
            <>
              {isPaused ? (
                <Button onClick={handleResume} variant="primary"><Play size={16} /> Resume</Button>
              ) : (
                <Button onClick={handlePause} variant="warning"><Pause size={16} /> Pause</Button>
              )}
              <Button onClick={handleStop} variant="danger"><Square size={16} /> Stop</Button>
            </>
          )}
        </div>
      </Card>

      {/* KPI Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <Card><Stat label="Total Reward" value={currentMetric?.reward.toFixed(2) || "0.00"} color={currentMetric && currentMetric.reward < -50 ? "text-red-400" : "text-green-400"} /></Card>
        <Card><Stat label="Wait Time" value={currentMetric?.wait_time.toFixed(1) || "0.0"} unit="s" color="text-yellow-400" subtitle="System average" /></Card>
        <Card><Stat label="Queue" value={currentMetric?.queue_length.toFixed(1) || "0"} unit="veh" color="text-orange-400" subtitle="Max intersection queue" /></Card>
        <Card><Stat label="Throughput" value={currentMetric?.throughput || "0"} unit="veh/h" color="text-blue-400" subtitle="Vehicles processed" /></Card>
        <Card><Stat label="Elapsed" value={elapsed} color="text-white" subtitle={`Episode ${currentMetric?.episode || 0}/${episodes} • Step ${currentMetric?.step || 0}`} /></Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 min-h-[400px]">
        <Card title="Reward History" icon={BarChart3} className="min-h-[350px]">
          <div className="flex-1 w-full h-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics}>
                <defs>
                  <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#4ecb71" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#4ecb71" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" vertical={false} />
                <XAxis dataKey="step" stroke="#8890a8" tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                <YAxis stroke="#8890a8" tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                <RechartsTooltip contentStyle={{ backgroundColor: '#1a1d27', borderColor: '#3a3f55', borderRadius: '8px', color: '#fff' }} />
                <Legend verticalAlign="top" height={36} iconType="circle" />
                <Area type="monotone" dataKey="reward" name="Reward" stroke="#4ecb71" strokeWidth={2} fillOpacity={1} fill="url(#colorReward)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Traffic Metrics" icon={Activity} className="min-h-[350px]">
          <div className="flex-1 w-full h-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" vertical={false} />
                <XAxis dataKey="step" stroke="#8890a8" tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" stroke="#facc15" tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                <YAxis yAxisId="right" orientation="right" stroke="#fb923c" tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                <RechartsTooltip contentStyle={{ backgroundColor: '#1a1d27', borderColor: '#3a3f55', borderRadius: '8px', color: '#fff' }} />
                <Legend verticalAlign="top" height={36} iconType="circle" />
                <Line yAxisId="left" type="monotone" dataKey="wait_time" name="Wait Time (s)" stroke="#facc15" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line yAxisId="right" type="monotone" dataKey="queue_length" name="Queue Length" stroke="#fb923c" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}
