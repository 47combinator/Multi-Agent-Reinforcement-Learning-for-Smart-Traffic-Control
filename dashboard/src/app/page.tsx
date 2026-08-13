"use client";

import React from "react";
import Link from "next/link";
import { PlayCircle, BarChart2, Activity, Info, ChevronRight, Zap } from "lucide-react";
import { Card, Button, Stat } from "@/components/ui";

export default function Home() {
  return (
    <div className="flex flex-col gap-8 max-w-7xl mx-auto w-full">
      <header className="flex flex-col gap-2">
        <h2 className="text-3xl font-bold tracking-tight text-white">Dashboard Overview</h2>
        <p className="text-[#8890a8] text-sm max-w-2xl">
          Welcome to the Smart Traffic Control Enterprise Dashboard. Monitor live telemetry, benchmark Reinforcement Learning algorithms against baseline models, and analyze intersection performance.
        </p>
      </header>

      {/* KPI Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card title="System Status" icon={Activity}>
          <div className="flex items-center gap-3 mt-2">
            <div className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e] animate-pulse" />
            <span className="text-2xl font-bold text-white">Online</span>
          </div>
          <p className="text-[#8890a8] text-sm mt-2">WebSocket server connected and ready for TraCI simulation.</p>
        </Card>
        
        <Card title="Available Models" icon={Zap}>
          <span className="text-3xl font-bold text-white mt-2 block">4</span>
          <p className="text-[#8890a8] text-sm mt-2">PPO, DQN, Q-Learning, and Fixed-Time baseline loaded.</p>
        </Card>

        <Card title="Total Intersections" icon={Info}>
          <span className="text-3xl font-bold text-white mt-2 block">1</span>
          <p className="text-[#8890a8] text-sm mt-2">Currently configured for the single intersection scenario.</p>
        </Card>
      </div>

      {/* Quick Actions */}
      <h3 className="text-xl font-bold tracking-tight text-white mt-4">Quick Actions</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="hover:border-blue-500/50 transition-colors group cursor-pointer">
          <Link href="/live" className="absolute inset-0 z-20" />
          <div className="flex flex-col h-full justify-between gap-6">
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400 group-hover:scale-110 transition-transform">
              <PlayCircle size={24} />
            </div>
            <div>
              <h4 className="text-lg font-bold text-white mb-2 flex items-center justify-between">
                Live Simulation
                <ChevronRight size={18} className="text-[#8890a8] group-hover:text-blue-400 group-hover:translate-x-1 transition-all" />
              </h4>
              <p className="text-[#8890a8] text-sm">Launch a new SUMO simulation instance and monitor the agent's live telemetry, rewards, and queue lengths in real-time.</p>
            </div>
          </div>
        </Card>

        <Card className="hover:border-purple-500/50 transition-colors group cursor-pointer">
          <Link href="/benchmark" className="absolute inset-0 z-20" />
          <div className="flex flex-col h-full justify-between gap-6">
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform">
              <BarChart2 size={24} />
            </div>
            <div>
              <h4 className="text-lg font-bold text-white mb-2 flex items-center justify-between">
                Run Benchmarks
                <ChevronRight size={18} className="text-[#8890a8] group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
              </h4>
              <p className="text-[#8890a8] text-sm">Compare the performance of all RL models side-by-side using advanced radar and bar charts to determine the optimal agent.</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
