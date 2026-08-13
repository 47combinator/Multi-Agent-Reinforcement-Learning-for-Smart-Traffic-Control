"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Home, PlayCircle, BarChart2, Info, Settings } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Live Simulation", href: "/live", icon: PlayCircle },
  { name: "Benchmark", href: "/benchmark", icon: BarChart2 },
  { name: "About", href: "/about", icon: Info },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-[#3a3f55]/50 bg-[#12141c]/80 backdrop-blur-3xl flex flex-col h-screen sticky top-0 shrink-0">
      <div className="p-6 flex items-center gap-3 border-b border-[#3a3f55]/30">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20 shrink-0">
          <Activity size={18} className="text-white" />
        </div>
        <div>
          <h1 className="font-bold text-[15px] leading-tight tracking-tight text-[#e8eaf0]">Smart Traffic</h1>
          <p className="text-[11px] text-[#8890a8] font-medium tracking-wide uppercase">Enterprise Edition</p>
        </div>
      </div>

      <nav className="flex-1 p-4 flex flex-col gap-1">
        <div className="text-xs font-semibold text-[#8890a8] mb-2 px-3 uppercase tracking-wider">Navigation</div>
        {links.map((link) => {
          const isActive = pathname === link.href;
          const Icon = link.icon;
          return (
            <Link
              key={link.name}
              href={link.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-blue-500/10 text-blue-400 font-semibold"
                  : "text-[#8890a8] hover:text-[#c8d0e0] hover:bg-white/5"
              )}
            >
              <Icon size={18} className={cn(isActive ? "text-blue-400" : "text-[#8890a8]")} />
              {link.name}
            </Link>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-[#3a3f55]/30">
        <div className="flex items-center gap-3 px-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center text-white text-xs font-bold">
            US
          </div>
          <div className="text-sm">
            <p className="text-[#e8eaf0] font-medium leading-none mb-1">User</p>
            <p className="text-[#8890a8] text-xs">Admin</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
