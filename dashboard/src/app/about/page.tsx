"use client";

import React from "react";
import { Info, Code, ExternalLink } from "lucide-react";
import { Card, Button } from "@/components/ui";

export default function About() {
  return (
    <div className="flex flex-col gap-8 max-w-4xl mx-auto w-full pb-10">
      <header>
        <h2 className="text-3xl font-bold tracking-tight text-white mb-2">About the Project</h2>
        <p className="text-[#8890a8] text-sm">
          Multi-Agent Reinforcement Learning for Smart Traffic Control.
        </p>
      </header>

      <Card title="Architecture & Methodology" icon={Info}>
        <div className="prose prose-invert max-w-none text-sm text-[#c8d0e0] leading-relaxed mt-2">
          <p className="mb-4">
            This enterprise application serves as the control center and visualization layer for an advanced Multi-Agent Reinforcement Learning (MARL) traffic management system. The core objective is to optimize traffic signal timings in urban intersections to minimize waiting times and maximize throughput.
          </p>
          <h4 className="text-white font-semibold mt-6 mb-2 text-base">Key Technologies</h4>
          <ul className="list-disc pl-5 mb-4 space-y-2">
            <li><strong>Environment:</strong> Eclipse SUMO (Simulation of Urban MObility) interfaced via TraCI.</li>
            <li><strong>Algorithms:</strong> Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and Q-Learning.</li>
            <li><strong>Frontend:</strong> Next.js 15, React, Tailwind CSS, Recharts, Framer Motion.</li>
            <li><strong>Backend:</strong> Python, Flask, Flask-SocketIO for real-time telemetry streaming.</li>
          </ul>
        </div>
      </Card>
      
      <div className="flex items-center gap-4">
        <Button onClick={() => window.open("https://github.com/47combinator/Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control")} className="bg-[#1a1d27] border border-[#3a3f55] hover:bg-[#2a2f45]">
          <Code size={18} /> View Source on GitHub
        </Button>
        <Button variant="outline">
          <ExternalLink size={18} /> Documentation
        </Button>
      </div>
    </div>
  );
}
