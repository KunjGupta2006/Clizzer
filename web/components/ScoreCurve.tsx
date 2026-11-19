"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceArea,
} from "recharts";
import { CurveData, ClipData } from "@/lib/api";

interface ScoreCurveProps {
  curve: CurveData;
  clips: ClipData[];
}

export default function ScoreCurve({ curve, clips }: ScoreCurveProps) {
  // Sample curve data down to ~200 points if large for responsive rendering
  const step = Math.max(1, Math.floor(curve.length / 200));
  const chartData = [];

  for (let i = 0; i < curve.length; i += step) {
    chartData.push({
      sec: i,
      fused: parseFloat((curve.fused[i] || 0).toFixed(2)),
      audio: parseFloat((curve.audio[i] || 0).toFixed(2)),
      chat: parseFloat((curve.chat[i] || 0).toFixed(2)),
    });
  }

  return (
    <div className="bg-[#14151d] border border-gray-800 rounded-2xl p-6 shadow-xl mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-white">Multimodal Signal Curve</h2>
          <p className="text-xs text-gray-400">
            Real-time fused score along with audio acoustics and chat velocity over time. Shaded areas represent extracted clip intervals.
          </p>
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis
              dataKey="sec"
              stroke="#4b5563"
              fontSize={11}
              tickFormatter={(val) => `${Math.floor(val / 60)}:${(val % 60).toString().padStart(2, "0")}`}
            />
            <YAxis stroke="#4b5563" fontSize={11} domain={[-3, 3]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0e0f15",
                borderColor: "#374151",
                borderRadius: "0.75rem",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />

            {/* Shaded Clip Regions */}
            {clips.map((clip, idx) => (
              <ReferenceArea
                key={idx}
                x1={Math.round(clip.start)}
                x2={Math.round(clip.end)}
                stroke="#ec4899"
                strokeOpacity={0.3}
                fill="#ec4899"
                fillOpacity={0.15}
              />
            ))}

            <Line
              type="monotone"
              dataKey="fused"
              stroke="#ec4899"
              strokeWidth={2.5}
              dot={false}
              name="Fused Viral Score"
            />
            <Line
              type="monotone"
              dataKey="audio"
              stroke="#a855f7"
              strokeWidth={1.5}
              dot={false}
              name="Audio Energy / Events"
            />
            <Line
              type="monotone"
              dataKey="chat"
              stroke="#06b6d4"
              strokeWidth={1.5}
              dot={false}
              name="Chat Velocity & Lexicons"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
