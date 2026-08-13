"use client";

import React, { useState } from "react";
import { Download, Award, BarChart2, CheckCircle2 } from "lucide-react";
import { Card, Button, Stat } from "@/components/ui";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis 
} from "recharts";
import { exportToJSON, exportToCSV } from "@/lib/export";
import { cn } from "@/lib/utils";

// Mock data based on realistic SUMO MARL benchmarks
const benchmarkResults = [
  { model: "PPO", reward: -152.3, wait_time: 45.2, queue_length: 3.1, throughput: 1120 },
  { model: "DQN", reward: -241.8, wait_time: 68.5, queue_length: 5.4, throughput: 1040 },
  { model: "Q-Learning", reward: -305.1, wait_time: 82.1, queue_length: 7.2, throughput: 980 },
  { model: "Fixed-Time", reward: -845.0, wait_time: 210.4, queue_length: 22.5, throughput: 850 },
];

const radarData = [
  { subject: 'Reward (Normalized)', PPO: 100, DQN: 80, QL: 65, FT: 20 },
  { subject: 'Wait Time (Inv)', PPO: 95, DQN: 75, QL: 60, FT: 10 },
  { subject: 'Queue (Inv)', PPO: 90, DQN: 70, QL: 55, FT: 15 },
  { subject: 'Throughput', PPO: 100, DQN: 85, QL: 75, FT: 50 },
];

export default function Benchmark() {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = (format: 'csv' | 'json') => {
    setIsExporting(true);
    setTimeout(() => {
      if (format === 'csv') exportToCSV(benchmarkResults, "benchmark_comparison");
      else exportToJSON(benchmarkResults, "benchmark_comparison");
      setIsExporting(false);
    }, 500);
  };

  return (
    <div className="flex flex-col gap-8 max-w-7xl mx-auto w-full pb-10">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white">Algorithm Benchmark</h2>
          <p className="text-[#8890a8] mt-1 text-sm">Performance comparison across 100 evaluation episodes (Seed: 42)</p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button variant="outline" onClick={() => handleExport('csv')} disabled={isExporting}>
            <Download size={16} /> Export CSV
          </Button>
          <Button variant="outline" onClick={() => handleExport('json')} disabled={isExporting}>
            <Download size={16} /> Export JSON
          </Button>
          <Button variant="outline" onClick={() => window.print()} disabled={isExporting}>
            <Download size={16} /> Export PDF
          </Button>
        </div>
      </header>

      {/* Winner Highlight */}
      <Card className="bg-gradient-to-r from-blue-900/40 to-indigo-900/40 border-blue-500/30">
        <div className="flex items-start gap-4">
          <div className="w-16 h-16 rounded-2xl bg-blue-500 flex items-center justify-center shadow-[0_0_30px_rgba(59,130,246,0.5)] shrink-0">
            <Award size={32} className="text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-white mb-1 flex items-center gap-2">
              PPO Agent is the Winner! <CheckCircle2 className="text-green-400" size={20} />
            </h3>
            <p className="text-[#c8d0e0] max-w-3xl leading-relaxed">
              Proximal Policy Optimization (PPO) outperformed all other models, reducing average waiting time by <span className="font-bold text-blue-400">78.5%</span> compared to the Fixed-Time baseline and <span className="font-bold text-blue-400">34%</span> compared to DQN. It successfully maintained the lowest queue lengths across all traffic phases.
            </p>
          </div>
        </div>
      </Card>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[450px]">
        {/* Radar Chart */}
        <Card title="Multi-Objective Performance (Normalized)" icon={BarChart2}>
          <div className="w-full h-full pb-8">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                <PolarGrid stroke="#2a2f45" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#8890a8', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="PPO" dataKey="PPO" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.4} />
                <Radar name="DQN" dataKey="DQN" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} />
                <Radar name="Q-Learning" dataKey="QL" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <RechartsTooltip contentStyle={{ backgroundColor: '#1a1d27', borderColor: '#3a3f55', borderRadius: '8px', color: '#fff' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Bar Chart */}
        <Card title="Waiting Time Comparison" icon={BarChart2}>
           <div className="w-full h-full pb-8">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmarkResults} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f45" vertical={false} />
                <XAxis dataKey="model" stroke="#8890a8" tickLine={false} axisLine={false} />
                <YAxis stroke="#8890a8" tickLine={false} axisLine={false} />
                <RechartsTooltip cursor={{fill: '#2a2f45'}} contentStyle={{ backgroundColor: '#1a1d27', borderColor: '#3a3f55', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="wait_time" name="Wait Time (s)" fill="#facc15" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Comparison Table */}
      <Card title="Detailed Results Matrix" className="overflow-x-auto">
        <table className="w-full text-left border-collapse mt-2">
          <thead>
            <tr className="border-b border-[#3a3f55]/50 text-[#8890a8] text-xs uppercase tracking-wider">
              <th className="pb-3 font-semibold">Model</th>
              <th className="pb-3 font-semibold text-right">Avg Reward</th>
              <th className="pb-3 font-semibold text-right">Wait Time (s)</th>
              <th className="pb-3 font-semibold text-right">Queue (veh)</th>
              <th className="pb-3 font-semibold text-right">Throughput</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {benchmarkResults.map((row, idx) => (
              <tr key={row.model} className={cn("border-b border-[#3a3f55]/20 last:border-0 hover:bg-white/5 transition-colors", idx === 0 && "bg-blue-500/5")}>
                <td className="py-4 font-semibold text-white flex items-center gap-2">
                  {idx === 0 && <Award size={16} className="text-blue-400" />} {row.model}
                </td>
                <td className={cn("py-4 text-right font-medium", idx === 0 ? "text-green-400" : "text-[#c8d0e0]")}>{row.reward.toFixed(1)}</td>
                <td className="py-4 text-right font-medium text-[#c8d0e0]">{row.wait_time.toFixed(1)}</td>
                <td className="py-4 text-right font-medium text-[#c8d0e0]">{row.queue_length.toFixed(1)}</td>
                <td className="py-4 text-right font-medium text-[#c8d0e0]">{row.throughput}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
