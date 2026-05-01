import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, TrendingUp, CheckCircle, Clock } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

// Format helper
const formatNumber = (num) => {
  if (num === null || num === undefined) return '0';
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
  return num.toString();
};

const MetricCard = ({ title, value, icon: Icon, trend, trendValue, colorClass, index }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
    <Card className="saas-card h-full">
      <CardContent className="p-6 flex flex-col h-full justify-between">
        <div className="flex items-center justify-between space-y-0 pb-2">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <div className={`p-2 rounded-lg ${colorClass}`}>
            <Icon className="h-4 w-4" />
          </div>
        </div>
        <div>
          <div className="text-3xl font-bold text-foreground">{value}</div>
          {trend && (
            <p className={`text-xs mt-2 flex items-center font-medium ${trend === 'up' ? 'text-emerald-500' : 'text-red-500'}`}>
              <TrendingUp className={`h-3 w-3 mr-1 ${trend === 'down' ? 'rotate-180' : ''}`} />
              {trendValue} from last week
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  </motion.div>
);

export default function Dashboard() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      fetchDashboardData();
      const interval = setInterval(() => fetchDashboardData(false), 5000);
      return () => clearInterval(interval);
    }
  }, [user]);

  const fetchDashboardData = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true);
      const [statsRes, alertsRes] = await Promise.all([
        api.get('/dashboard/stats'),
        api.get('/alerts')
      ]);
      setStats(statsRes.data);
      setAlerts(Array.isArray(alertsRes.data) ? alertsRes.data.slice(0, 5) : []);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  if (loading && !stats) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
        </div>
      </Layout>
    );
  }

  // Normalize data for charts
  const performanceData = Array.isArray(stats?.accuracy_trend) ? stats.accuracy_trend.map(item => ({
    time: new Date(item.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    accuracy: item.accuracy ? (item.accuracy * 100).toFixed(1) : 0,
  })) : [];

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Overview</h1>
          <p className="text-muted-foreground mt-1">Monitor the health and performance of your ML models.</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Models"
            value={formatNumber(stats?.total_models)}
            icon={Activity}
            colorClass="bg-primary/10 text-primary"
            index={0}
          />
          <MetricCard
            title="Total Predictions"
            value={formatNumber(stats?.total_predictions)}
            icon={TrendingUp}
            trend="up"
            trendValue="+12%"
            colorClass="bg-emerald-500/10 text-emerald-500"
            index={1}
          />
          <MetricCard
            title="Active Alerts"
            value={formatNumber(stats?.active_alerts)}
            icon={AlertTriangle}
            trend="down"
            trendValue="-2"
            colorClass="bg-rose-500/10 text-rose-500"
            index={2}
          />
          <MetricCard
            title="Avg. Accuracy"
            value={`${stats?.avg_accuracy ? (stats.avg_accuracy * 100).toFixed(1) : '0'}%`}
            icon={CheckCircle}
            trend="up"
            trendValue="+1.2%"
            colorClass="bg-blue-500/10 text-blue-500"
            index={3}
          />
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-7">
          <Card className="saas-card lg:col-span-4">
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
              <CardDescription>Accuracy trend over the last 30 days</CardDescription>
            </CardHeader>
            <CardContent>
              {performanceData.length > 0 ? (
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={performanceData}>
                      <defs>
                        <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#4f46e5" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#4f46e5" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                      <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px', color: 'hsl(var(--foreground))' }}
                        itemStyle={{ color: 'hsl(var(--primary))' }}
                      />
                      <Area type="monotone" dataKey="accuracy" stroke="#4f46e5" strokeWidth={2} fillOpacity={1} fill="url(#colorAccuracy)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm border-2 border-dashed border-border rounded-xl">
                  No performance data available
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="saas-card lg:col-span-3">
            <CardHeader>
              <CardTitle>Recent Alerts</CardTitle>
              <CardDescription>Latest anomalies and drift warnings</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.length > 0 ? alerts.map((alert, i) => (
                  <motion.div
                    key={alert.id || i}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-center gap-4 p-3 rounded-xl border border-border bg-background"
                  >
                    <div className={`p-2 rounded-full flex-shrink-0 ${
                      alert.severity === 'high' ? 'bg-rose-500/10 text-rose-500' :
                      alert.severity === 'medium' ? 'bg-amber-500/10 text-amber-500' :
                      'bg-blue-500/10 text-blue-500'
                    }`}>
                      <AlertTriangle className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {alert.message || alert.type}
                      </p>
                      <div className="flex items-center text-xs text-muted-foreground mt-1">
                        <Clock className="mr-1 h-3 w-3" />
                        {new Date(alert.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </motion.div>
                )) : (
                  <div className="text-center py-10">
                    <CheckCircle className="mx-auto h-8 w-8 text-emerald-500/50 mb-3" />
                    <p className="text-sm text-muted-foreground">All systems normal. No active alerts.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
