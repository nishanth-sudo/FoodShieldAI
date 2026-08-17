import { useEffect, useState } from "react";
import { useLanguage } from "../context/LanguageContext";

interface Props {
  score: number;
  label: string;
}

export function FreshnessGauge({ score, label }: Props) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const { t } = useLanguage();

  // Animating the score count
  useEffect(() => {
    const duration = 800; // ms
    const start = 0;
    const end = score;
    if (start === end) return;

    let startTime: number | null = null;
    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = timestamp - startTime;
      const progressPercentage = Math.min(progress / duration, 1);
      setAnimatedScore(Math.floor(progressPercentage * (end - start) + start));

      if (progressPercentage < 1) {
        requestAnimationFrame(animate);
      }
    };
    requestAnimationFrame(animate);
  }, [score]);

  // Color selection
  const strokeColor = score >= 80 ? "#22c55e" : score >= 50 ? "#eab308" : "#ef4444";
  const textColorClass = score >= 80 ? "text-green-600 dark:text-green-400" : score >= 50 ? "text-amber-500" : "text-red-500";
  const bgBadgeClass = score >= 80 ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800" : score >= 50 ? "bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800" : "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800";

  // SVG parameters
  const radius = 50;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col items-center justify-center text-center">
      <p className="text-sm font-bold text-gray-500 dark:text-gray-400 mb-4 uppercase tracking-wider">{label}</p>
      
      <div className="relative flex items-center justify-center w-36 h-36">
        <svg className="w-full h-full transform -rotate-90">
          {/* Background circle */}
          <circle
            cx="72"
            cy="72"
            r={radius}
            stroke="#f3f4f6"
            className="dark:stroke-gray-700"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          {/* Progress circle */}
          <circle
            cx="72"
            cy="72"
            r={radius}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-300 ease-out"
          />
        </svg>
        <div className="absolute flex flex-col items-center justify-center">
          <span className={`text-3xl font-extrabold tracking-tight ${textColorClass}`}>{animatedScore}%</span>
          <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">{t.score}</span>
        </div>
      </div>

      <div className={`mt-4 px-3 py-1 rounded-full text-xs font-bold border ${bgBadgeClass}`}>
        {score >= 80 ? t.freshSafe : score >= 50 ? t.cautionWarning : t.spoiledDefective}
      </div>
    </div>
  );
}
