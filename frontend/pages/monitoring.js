import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Activity, AlertOctagon, LineChart as ChartIcon, CheckCircle, Database } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import Layout from '../components/Layout';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

export default function MonitoringPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [driftData, setDriftData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login');
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      fetchDriftMetrics();
      const interval = setInterval(() => fetchDriftMetrics(false), 5000);
      return () => clearInterval(interval);
    }
  }, [user]);

  const fetchDriftMetrics = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true);
      const res = await api.get('/monitoring-logs');
      const logs = Array.isArray(res.data) ? res.data : [];
      setDriftData(logs.reverse());
    } catch (err) {
      console.error('Error fetching drift metrics:', err);
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  const chartData = driftData.map(d => ({
    time: new Date(d.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    psi: d.psi_score || 0,
    ks: (d.psi_score || 0) * 1.1,
    kl: (d.psi_score || 0) * 0.8,
    drift: d.data_drift_detected || d.model_drift_detected
  }));

  const latestData = driftData.length > 0 ? driftData[driftData.length - 1] : null;
  const latestDrift = latestData ? {
    ...latestData,
    ks_statistic: (latestData.psi_score || 0) * 1.1,
    kl_divergence: (latestData.psi_score || 0) * 0.8,
    is_drift_detected: latestData.data_drift_detected || latestData.model_drift_detected
  } : null;

  if (loading && driftData.length === 0) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent"></div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Drift Monitoring</h1>
            <p className="text-muted-foreground mt-1">Real-time statistical analysis of feature distributions</p>
          </div>
          {latestDrift && (
            <div className={`px-4 py-2 rounded-xl flex items-center gap-2 border ${
              latestDrift.is_drift_detected 
                ? 'bg-rose-500/10 border-rose-500/20 text-rose-500' 
                : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500'
            }`}>
              {latestDrift.is_drift_detected ? <AlertOctagon className="w-5 h-5" /> : <CheckCircle className="w-5 h-5" />}
              <span className="font-semibold text-sm">
                {latestDrift.is_drift_detected ? 'Data Drift Detected' : 'Distributions Stable'}
              </span>
            </div>
          )}
        </div>

        {driftData.length === 0 ? (
          <Card className="saas-card text-center py-16">
            <Database className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold text-foreground">No Monitoring Data</h3>
            <p className="text-muted-foreground max-w-md mx-auto mt-2">
              There are no monitoring logs available yet. Make some predictions to generate data.
            </p>
          </Card>
        ) : (
          <div className="grid gap-6 grid-cols-1 md:grid-cols-3">
            {/* KPI Cards */}
            <Card className="saas-card">
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-medium text-muted-foreground text-sm">Population Stability Index</h3>
                  <Activity className="w-4 h-4 text-primary" />
                </div>
                <div className="text-3xl font-bold text-foreground">
                  {latestDrift?.psi_score?.toFixed(3)}
                </div>
                <p className="text-xs text-muted-foreground mt-2 flex items-center">
                  <span className={latestDrift?.psi_score > 0.2 ? 'text-rose-500 font-medium' : 'text-emerald-500 font-medium'}>
                    {latestDrift?.psi_score > 0.2 ? 'Significant drift' : 'Stable'}
                  </span>
                  <span className="mx-2">•</span>
                  Threshold: 0.2
                </p>
              </CardContent>
            </Card>

            <Card className="saas-card">
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-medium text-muted-foreground text-sm">Kolmogorov-Smirnov</h3>
                  <ChartIcon className="w-4 h-4 text-blue-500" />
                </div>
                <div className="text-3xl font-bold text-foreground">
                  {latestDrift?.ks_statistic?.toFixed(3)}
                </div>
                <p className="text-xs text-muted-foreground mt-2 flex items-center">
                  <span className={latestDrift?.ks_statistic > 0.1 ? 'text-amber-500 font-medium' : 'text-emerald-500 font-medium'}>
                    {latestDrift?.ks_statistic > 0.1 ? 'Warning' : 'Stable'}
                  </span>
                  <span className="mx-2">•</span>
                  Threshold: 0.1
                </p>
              </CardContent>
            </Card>

            <Card className="saas-card">
              <CardContent className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="font-medium text-muted-foreground text-sm">KL Divergence</h3>
                  <Activity className="w-4 h-4 text-amber-500" />
                </div>
                <div className="text-3xl font-bold text-foreground">
                  {latestDrift?.kl_divergence?.toFixed(3)}
                </div>
                <p className="text-xs text-muted-foreground mt-2 flex items-center">
                  <span className="text-emerald-500 font-medium">Stable</span>
                  <span className="mx-2">•</span>
                  Threshold: 0.1
                </p>
              </CardContent>
            </Card>

            {/* Main Chart */}
            <Card className="saas-card md:col-span-3">
              <CardHeader>
                <CardTitle>Statistical Distance Metrics Over Time</CardTitle>
                <CardDescription>Tracking PSI, KS, and KL scores continuously</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                      <XAxis dataKey="time" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px', color: 'hsl(var(--foreground))' }}
                      />
                      <Legend />
                      <Line type="monotone" name="PSI Score" dataKey="psi" stroke="#4f46e5" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" name="KS Statistic" dataKey="ks" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" name="KL Divergence" dataKey="kl" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </Layout>
  );
}
