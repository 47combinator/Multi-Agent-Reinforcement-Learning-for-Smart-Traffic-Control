import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export const Button = ({ children, className, variant = "primary", ...props }: any) => {
  const variants: any = {
    primary: "bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(37,99,235,0.4)] border border-blue-500/50",
    danger: "bg-red-600 hover:bg-red-500 text-white shadow-[0_0_15px_rgba(220,38,38,0.4)] border border-red-500/50",
    warning: "bg-yellow-600 hover:bg-yellow-500 text-white shadow-[0_0_15px_rgba(202,138,4,0.4)] border border-yellow-500/50",
    outline: "bg-transparent border border-[#3a3f55]/50 hover:border-[#3a3f55] hover:bg-white/5 text-white",
  };
  return (
    <button
      className={cn("px-4 py-2 rounded-xl font-medium transition-all duration-300 flex items-center justify-center gap-2", variants[variant], className)}
      {...props}
    >
      {children}
    </button>
  );
};

export const Card = ({ children, className, title, icon: Icon, action }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 15 }}
    animate={{ opacity: 1, y: 0 }}
    className={cn("bg-[#1a1d27]/80 backdrop-blur-xl border border-[#3a3f55]/50 rounded-2xl p-5 shadow-2xl relative overflow-hidden", className)}
  >
    <div className="absolute -top-24 -right-24 w-48 h-48 bg-blue-500/5 blur-[80px] rounded-full pointer-events-none" />
    
    {(title || action) && (
      <div className="flex items-center justify-between mb-4 relative z-10">
        <div className="flex items-center gap-2 text-[#c8d0e0]">
          {Icon && <Icon size={18} className="text-blue-400" />}
          <h3 className="font-semibold text-sm tracking-wide uppercase">{title}</h3>
        </div>
        {action && <div>{action}</div>}
      </div>
    )}
    <div className="relative z-10 h-full flex flex-col">{children}</div>
  </motion.div>
);

export const Stat = ({ label, value, unit, color = "text-white", subtitle }: any) => (
  <div className="flex flex-col">
    <span className="text-[#8890a8] text-xs font-semibold uppercase tracking-wider mb-1">{label}</span>
    <div className="flex items-baseline gap-1">
      <span className={cn("text-3xl font-bold tracking-tight", color)}>{value}</span>
      {unit && <span className="text-[#8890a8] text-sm font-medium">{unit}</span>}
    </div>
    {subtitle && <span className="text-xs text-[#8890a8] mt-1">{subtitle}</span>}
  </div>
);
