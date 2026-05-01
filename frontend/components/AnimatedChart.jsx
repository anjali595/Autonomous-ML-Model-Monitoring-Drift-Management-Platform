import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export function AnimatedLineChart({ data, color = '#3b82f6', gradientId = 'lineGradient', ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, type: 'spring' }}
      className="bg-gradient-to-br from-blue-900/60 to-gray-900/60 rounded-2xl p-4 shadow-xl backdrop-blur-xl"
    >
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} {...props}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={color} stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" stroke="#64748b" />
          <YAxis stroke="#64748b" />
          <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 12, color: '#fff' }} />
          <Line type="monotone" dataKey="accuracy" stroke={`url(#${gradientId})`} strokeWidth={3} dot={false} activeDot={{ r: 7 }} />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

export function AnimatedAreaChart({ data, color = '#6366f1', gradientId = 'areaGradient', ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, type: 'spring' }}
      className="bg-gradient-to-br from-blue-900/60 to-gray-900/60 rounded-2xl p-4 shadow-xl backdrop-blur-xl"
    >
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} {...props}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.7}/>
              <stop offset="95%" stopColor={color} stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" stroke="#64748b" />
          <YAxis stroke="#64748b" />
          <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 12, color: '#fff' }} />
          <Area type="monotone" dataKey="drift" stroke={color} fill={`url(#${gradientId})`} strokeWidth={3} activeDot={{ r: 7 }} />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

export function AnimatedBarChart({ data, color = '#0fffc1', ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, type: 'spring' }}
      className="bg-gradient-to-br from-blue-900/60 to-gray-900/60 rounded-2xl p-4 shadow-xl backdrop-blur-xl"
    >
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} {...props}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" stroke="#64748b" />
          <YAxis stroke="#64748b" />
          <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: 12, color: '#fff' }} />
          <Bar dataKey="value" fill={color} radius={[12, 12, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
