import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Smart Traffic Control - Enterprise Dashboard",
  description: "Live MARL agent telemetry and controls",
};

import Sidebar from "@/components/Sidebar";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex text-[#e8eaf0] bg-[#0f1117] selection:bg-blue-500/30">
        <Sidebar />
        <div className="flex-1 flex flex-col h-screen overflow-y-auto relative">
          {/* Background ambient light */}
          <div className="fixed top-0 left-1/4 w-1/2 h-64 bg-blue-600/10 blur-[120px] rounded-full pointer-events-none z-0" />
          <div className="relative z-10 p-8">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
