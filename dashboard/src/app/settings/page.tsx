"use client";

import React, { useState } from "react";
import { Settings as SettingsIcon, Save } from "lucide-react";
import { Card, Button } from "@/components/ui";

export default function Settings() {
  const [chartRefresh, setChartRefresh] = useState("Fast (60Hz)");
  const [dataRetention, setDataRetention] = useState("1000");

  return (
    <div className="flex flex-col gap-8 max-w-4xl mx-auto w-full pb-10">
      <header>
        <h2 className="text-3xl font-bold tracking-tight text-white mb-2">Settings</h2>
        <p className="text-[#8890a8] text-sm">
          Configure application preferences and chart rendering limits.
        </p>
      </header>

      <Card title="Performance Preferences" icon={SettingsIcon}>
        <div className="flex flex-col gap-6 mt-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-white font-medium text-sm">Chart Refresh Rate</h4>
              <p className="text-[#8890a8] text-xs mt-1">Adjust how frequently the live charts re-render.</p>
            </div>
            <select 
              value={chartRefresh}
              onChange={(e) => setChartRefresh(e.target.value)}
              className="bg-[#12141c] border border-[#3a3f55]/50 text-sm rounded-lg px-3 py-2 text-white outline-none focus:border-blue-500 w-40"
            >
              <option>Realtime (120Hz)</option>
              <option>Fast (60Hz)</option>
              <option>Smooth (30Hz)</option>
            </select>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-white font-medium text-sm">Data Retention Limit</h4>
              <p className="text-[#8890a8] text-xs mt-1">Maximum number of data points to keep in memory for live charts.</p>
            </div>
            <select 
              value={dataRetention}
              onChange={(e) => setDataRetention(e.target.value)}
              className="bg-[#12141c] border border-[#3a3f55]/50 text-sm rounded-lg px-3 py-2 text-white outline-none focus:border-blue-500 w-40"
            >
              <option value="500">500 points</option>
              <option value="1000">1000 points</option>
              <option value="5000">5000 points</option>
            </select>
          </div>
        </div>
      </Card>
      
      <div className="flex justify-end">
        <Button onClick={() => alert("Settings saved!")}><Save size={16} /> Save Preferences</Button>
      </div>
    </div>
  );
}
