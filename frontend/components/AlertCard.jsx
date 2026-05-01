import { motion } from 'framer-motion';
import { AlertTriangle } from 'lucide-react';

const severityColors = {
  critical: 'bg-red-500 text-white',
  warning: 'bg-yellow-400 text-black',
  normal: 'bg-green-500 text-white',
};

export default function AlertCard({ title, message, severity = 'normal', time = 'now' }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, type: 'spring' }}
      whileHover={{ scale: 1.03, boxShadow: '0 0 16px 2px #f87171' }}
      className={`relative flex items-center gap-4 p-4 rounded-2xl shadow-lg glass-card bg-gradient-to-br from-blue-900/60 to-gray-900/60 backdrop-blur-xl border-l-4 ${severity === 'critical' ? 'border-red-500' : severity === 'warning' ? 'border-yellow-400' : 'border-green-500'} mb-3`}
    >
      <span className={`inline-flex items-center justify-center rounded-full w-10 h-10 ${severityColors[severity]} animate-pulse`}>
        <AlertTriangle className="w-6 h-6" />
      </span>
      <div className="flex-1">
        <div className="font-semibold text-white/90">{title}</div>
        <div className="text-sm text-white/70">{message}</div>
      </div>
      <span className="text-xs text-white/60 ml-2">{time}</span>
    </motion.div>
  );
}
